import torch
import torch.nn as nn
from stable_baselines3.sac.policies import SACPolicy, Actor, ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DummyExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1)
    def forward(self, observations):
        return torch.zeros((observations["nodes"].shape[0], 1), device=observations["nodes"].device)

# 🚀 顶会架构：将 GCN 升级为 Transformer，打通 1-10 关节的信息瓶颈！
class JointTokenTransformer(nn.Module):
    def __init__(self, node_feat_dim=6, global_feat_dim=4, hidden_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        self.node_emb = nn.Linear(node_feat_dim, hidden_dim)
        self.global_emb = nn.Linear(global_feat_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, nodes, mask, global_feat):
        B, N, _ = nodes.shape
        x = self.node_emb(nodes) 
        g = self.global_emb(global_feat).unsqueeze(1) 
        x = x + g  # 将全局目标广播给所有关节
        
        # 🚨 PyTorch Transformer 的 mask 逻辑：填充位(废弃关节)为 True
        padding_mask = (mask == 0).bool() 
        
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        return out

class MaskedGraphActor(Actor):
    def __init__(self, observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs):
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs)
        
        hidden_dim = 256
        self.max_j = 10
        self.node_dim = 6 
        self.global_dim = 4
        
        self.backbone = JointTokenTransformer(self.node_dim, self.global_dim, hidden_dim)
        
        combined_dim = hidden_dim + self.node_dim + self.global_dim
        
        self.action_mean = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.GELU(), 
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1)
        )
        self.action_log_std = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.GELU(), 
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1)
        )

    def get_action_dist_params(self, obs):
        nodes, mask, global_feat = obs["nodes"], obs["mask"].float(), obs["global"]
        B, N, _ = nodes.shape
        
        h_nodes = self.backbone(nodes, mask, global_feat)
        global_expanded = global_feat.unsqueeze(1).expand(B, N, -1)
        combined_h = torch.cat([h_nodes, nodes, global_expanded], dim=-1)
        
        mean = self.action_mean(combined_h).squeeze(-1)
        log_std = torch.clamp(self.action_log_std(combined_h).squeeze(-1), -20, 2)
        
        mean = mean * mask
        return mean, log_std, {}

# 🚀 全局上帝视角 Critic
class MaskedGraphCritic(ContinuousCritic):
    def __init__(self, observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs):
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs)
        
        hidden_dim = 256
        self.max_j = 10
        self.node_dim = 6
        self.global_dim = 4
        self.action_dim = action_space.shape[0]
        
        self.q_networks = nn.ModuleList()
        for _ in range(self.n_critics):
            backbone = JointTokenTransformer(self.node_dim, self.global_dim, hidden_dim)
            # 🚨 全局特征 (hidden_dim) + 目标 (4) + 动作 (10)
            combined_dim = hidden_dim + self.global_dim + self.action_dim
            
            head = nn.Sequential(
                nn.Linear(combined_dim, 256), nn.ReLU(), 
                nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)
            )
            self.q_networks.append(nn.ModuleDict({"backbone": backbone, "head": head}))

    def forward(self, obs, actions):
        nodes, mask, global_feat = obs["nodes"], obs["mask"].float(), obs["global"]
        masked_actions = actions * mask 
        
        q_values = []
        for q_net in self.q_networks:
            h_nodes = q_net["backbone"](nodes, mask, global_feat)
            
            # ✅ 全局池化：把所有关节浓缩为 1 个代表整体的特征
            h_graph = (h_nodes * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
            
            # 拼装给上帝视角打分
            c = torch.cat([h_graph, global_feat, masked_actions], dim=-1)
            q = q_net["head"](c)
            q_values.append(q)
            
        return tuple(q_values)

    def q1_forward(self, obs, actions):
        return self.forward(obs, actions)[0]

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