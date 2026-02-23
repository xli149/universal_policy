import torch
import torch.nn as nn
from stable_baselines3.sac.policies import SACPolicy, Actor, ContinuousCritic
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ✅ 伪装特征提取器 (保持不变)
class DummyExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1)
    def forward(self, observations):
        return torch.zeros((observations["nodes"].shape[0], 1), device=observations["nodes"].device)

# ✅ Transformer 骨干网络 (保持不变)
class JointTokenTransformer(nn.Module):
    def __init__(self, node_feat_dim=5, global_feat_dim=4, max_joints=10, d_model=64):
        super().__init__()
        self.node_in = nn.Linear(node_feat_dim, d_model)
        self.global_in = nn.Linear(global_feat_dim, d_model)
        self.pos_emb = nn.Embedding(max_joints + 1, d_model)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128, 
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

    def forward(self, nodes, mask, global_feat):
        B, N, _ = nodes.shape
        x = self.node_in(nodes)
        pos = torch.arange(N, device=nodes.device).unsqueeze(0).expand(B, N)
        x = x + self.pos_emb(pos) + self.global_in(global_feat).unsqueeze(1)
        return self.encoder(x, src_key_padding_mask=(mask < 0.5))

# 🚀 1. 定制 SAC Actor
class MaskedGraphActor(Actor):
    def __init__(self, observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs):
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs)
        
        d_model = 64
        self.max_j = 10
        self.node_dim = 5
        self.backbone = JointTokenTransformer(self.node_dim, 4, self.max_j, d_model)
        combined_dim = d_model + self.node_dim
        
        # SAC 的 Actor 需要输出 mean 和 log_std
        self.action_mean = nn.Sequential(nn.Linear(combined_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.action_log_std = nn.Sequential(nn.Linear(combined_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def get_action_dist_params(self, obs):
        nodes, mask, global_feat = obs["nodes"], obs["mask"].float(), obs["global"]
        
        h_nodes = self.backbone(nodes, mask, global_feat)
        combined_h = torch.cat([h_nodes, nodes], dim=-1)
        
        mean = self.action_mean(combined_h).squeeze(-1)
        log_std = self.action_log_std(combined_h).squeeze(-1)
        
        # 限制方差防止数值崩溃 (SAC 的标准保护)
        log_std = torch.clamp(log_std, -20, 2)
        
        # 掩盖掉不存在的关节
        mean = mean * mask
        log_std = log_std * mask + (-20.0) * (1.0 - mask)
        return mean, log_std, {}

# 🚀 2. 定制 SAC Critic (双 Q 网络)
class MaskedGraphCritic(ContinuousCritic):
    def __init__(self, observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs):
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs)
        
        d_model = 64
        self.max_j = 10
        self.node_dim = 5
        
        # 创建两个独立的 Q 网络
        self.q_networks = nn.ModuleList()
        for _ in range(self.n_critics):
            backbone = JointTokenTransformer(self.node_dim, 4, self.max_j, d_model)
            # 注意：Critic 的输入拼接了动作维度 (+1)
            head = nn.Sequential(nn.Linear(d_model + self.node_dim + 1, 64), nn.ReLU(), nn.Linear(64, 1))
            self.q_networks.append(nn.ModuleDict({"backbone": backbone, "head": head}))

    def forward(self, obs, actions):
        nodes, mask, global_feat = obs["nodes"], obs["mask"].float(), obs["global"]
        # 将动作切分到每个节点上：(B, N) -> (B, N, 1)
        actions_unsq = actions.unsqueeze(-1)
        
        q_values = []
        for q_net in self.q_networks:
            h = q_net["backbone"](nodes, mask, global_feat)
            # 将 (Transformer特征 + 原始状态 + 动作) 拼在一起评估 Q 值！
            c = torch.cat([h, nodes, actions_unsq], dim=-1)
            q_nodes = q_net["head"](c).squeeze(-1)
            q = (q_nodes * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            q_values.append(q.view(-1, 1))
            
        return tuple(q_values)

    def q1_forward(self, obs, actions):
        return self.forward(obs, actions)[0]

# 🚀 3. 打包成 SB3 兼容的 Policy
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