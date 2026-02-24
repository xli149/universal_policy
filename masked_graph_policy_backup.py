import torch
import torch.nn as nn
from stable_baselines3.sac.policies import SACPolicy, Actor, ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 伪装特征提取器
class DummyExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1)
    def forward(self, observations):
        return torch.zeros((observations["nodes"].shape[0], 1), device=observations["nodes"].device)

# 瘦身版 Transformer 骨干
class JointTokenTransformer(nn.Module):
    def __init__(self, node_feat_dim=5, global_feat_dim=4, max_joints=10, d_model=32):
        super().__init__()
        self.node_in = nn.Linear(node_feat_dim, d_model)
        self.global_in = nn.Linear(global_feat_dim, d_model)
        self.pos_emb = nn.Embedding(max_joints + 1, d_model)
        
        # 降低层数和头数，加速训练并减少过拟合
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=2, dim_feedforward=64, 
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

    def forward(self, nodes, mask, global_feat):
        B, N, _ = nodes.shape
        x = self.node_in(nodes)
        pos = torch.arange(N, device=nodes.device).unsqueeze(0).expand(B, N)
        x = x + self.pos_emb(pos) + self.global_in(global_feat).unsqueeze(1)
        return self.encoder(x, src_key_padding_mask=(mask < 0.5))

# 定制 SAC Actor
class MaskedGraphActor(Actor):
    def __init__(self, observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs):
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs)
        
        d_model = 32
        self.max_j = 10
        self.node_dim = 5
        self.global_dim = 4
        self.backbone = JointTokenTransformer(self.node_dim, self.global_dim, self.max_j, d_model)
        
        # 引入全局残差：将 Transformer 特征、原始节点特征、全局特征拼在一起
        combined_dim = d_model + self.node_dim + self.global_dim
        
        self.action_mean = nn.Sequential(nn.Linear(combined_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.action_log_std = nn.Sequential(nn.Linear(combined_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def get_action_dist_params(self, obs):
        nodes, mask, global_feat = obs["nodes"], obs["mask"].float(), obs["global"]
        B, N, _ = nodes.shape
        
        h_nodes = self.backbone(nodes, mask, global_feat)
        
        # 将 global_feat 扩展到每个节点维度进行拼接
        global_expanded = global_feat.unsqueeze(1).expand(B, N, -1)
        combined_h = torch.cat([h_nodes, nodes, global_expanded], dim=-1)
        
        mean = self.action_mean(combined_h).squeeze(-1)
        log_std = self.action_log_std(combined_h).squeeze(-1)
        
        log_std = torch.clamp(log_std, -20, 2)
        
        mean = mean * mask
        log_std = log_std * mask + (-20.0) * (1.0 - mask)
        return mean, log_std, {}

# 定制 SAC Critic
class MaskedGraphCritic(ContinuousCritic):
    def __init__(self, observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs):
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs)
        
        d_model = 32
        self.max_j = 10
        self.node_dim = 5
        self.global_dim = 4
        self.action_dim = action_space.shape[0]
        
        self.q_networks = nn.ModuleList()
        for _ in range(self.n_critics):
            backbone = JointTokenTransformer(self.node_dim, self.global_dim, self.max_j, d_model)
            # 拼接: Transformer(d_model) + Node(5) + Global(4) + Action(action_dim)
            combined_dim = d_model + self.node_dim + self.global_dim + self.action_dim
            head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            self.q_networks.append(nn.ModuleDict({"backbone": backbone, "head": head}))

    def forward(self, obs, actions):
        nodes, mask, global_feat = obs["nodes"], obs["mask"].float(), obs["global"]
        B, N, _ = nodes.shape
        
        # 【关键修复】动作是 (B, action_dim)，需广播到所有节点 (B, N, action_dim)
        actions_expanded = actions.unsqueeze(1).expand(B, N, -1)
        global_expanded = global_feat.unsqueeze(1).expand(B, N, -1)
        
        q_values = []
        for q_net in self.q_networks:
            h = q_net["backbone"](nodes, mask, global_feat)
            c = torch.cat([h, nodes, global_expanded, actions_expanded], dim=-1)
            q_nodes = q_net["head"](c).squeeze(-1)
            # 对所有有效节点求平均 Q 值
            q = (q_nodes * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            q_values.append(q.view(-1, 1))
            
        return tuple(q_values)

    def q1_forward(self, obs, actions):
        return self.forward(obs, actions)[0]

# 打包为 SB3 Policy
class MaskedGraphSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = DummyExtractor
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return MaskedGraphActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return MaskedGraphCritic(**critic_kwargs).to(self.device)