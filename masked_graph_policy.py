import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.sac.policies import SACPolicy, Actor, ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 1. 伪装特征提取器
class DummyExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1)
    def forward(self, observations):
        return torch.zeros((observations["nodes"].shape[0], 1), device=observations["nodes"].device)

# 2. GCN 骨干网络 
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        out = torch.bmm(adj, x)
        out = self.linear(out)
        return F.relu(out)

class JointGraphGCN(nn.Module):
    def __init__(self, node_feat_dim=5, global_feat_dim=4, max_joints=10, hidden_dim=128):
        super().__init__()
        self.max_joints = max_joints
        
        self.node_mlp = nn.Linear(node_feat_dim, hidden_dim)
        self.global_mlp = nn.Linear(global_feat_dim, hidden_dim)
        
        self.gcn1 = GraphConvLayer(hidden_dim, hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim)

        adj = torch.zeros(max_joints, max_joints)
        for i in range(max_joints):
            adj[i, i] = 1.0  
            if i > 0: adj[i, i-1] = 1.0  
            if i < max_joints - 1: adj[i, i+1] = 1.0  
        
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        self.register_buffer('norm_adj', norm_adj)

    def forward(self, nodes, mask, global_feat):
        B, N, _ = nodes.shape
        x = self.node_mlp(nodes) 
        g = self.global_mlp(global_feat).unsqueeze(1) 
        x = x + g  
        
        adj_batch = self.norm_adj.unsqueeze(0).expand(B, N, N).clone()
        mask_matrix = mask.unsqueeze(-1) * mask.unsqueeze(1)
        adj_batch = adj_batch * mask_matrix
        
        x = self.gcn1(x, adj_batch)
        x = self.gcn2(x, adj_batch)
        return x

# 3. 定制 SAC Actor
class MaskedGraphActor(Actor):
    def __init__(self, observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs):
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs)
        
        hidden_dim = 128
        self.max_j = 10
        self.node_dim = 5
        self.global_dim = 4
        
        self.backbone = JointGraphGCN(self.node_dim, self.global_dim, self.max_j, hidden_dim)
        
        combined_dim = hidden_dim + self.node_dim + self.global_dim
        self.action_mean = nn.Sequential(nn.Linear(combined_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.action_log_std = nn.Sequential(nn.Linear(combined_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def get_action_dist_params(self, obs):
        nodes, mask, global_feat = obs["nodes"], obs["mask"].float(), obs["global"]
        B, N, _ = nodes.shape
        
        h_nodes = self.backbone(nodes, mask, global_feat)
        
        global_expanded = global_feat.unsqueeze(1).expand(B, N, -1)
        combined_h = torch.cat([h_nodes, nodes, global_expanded], dim=-1)
        
        mean = self.action_mean(combined_h).squeeze(-1)
        log_std = self.action_log_std(combined_h).squeeze(-1)
        log_std = torch.clamp(log_std, -20, 2)
        
        # 🚀 修改 1：只锁死均值，释放方差！
        mean = mean * mask
        # (删除了 log_std = log_std * mask + ...) 
        # 让假关节的 log_std 自由输出，从而喂饱 SAC 的 auto entropy 调节器
        
        return mean, log_std, {}

# 4. 定制 SAC Critic
class MaskedGraphCritic(ContinuousCritic):
    def __init__(self, observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs):
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs)
        
        hidden_dim = 128
        self.max_j = 10
        self.node_dim = 5
        self.global_dim = 4
        self.action_dim = action_space.shape[0]
        
        self.q_networks = nn.ModuleList()
        for _ in range(self.n_critics):
            backbone = JointGraphGCN(self.node_dim, self.global_dim, self.max_j, hidden_dim)
            combined_dim = hidden_dim + self.node_dim + self.global_dim + self.action_dim
            head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            self.q_networks.append(nn.ModuleDict({"backbone": backbone, "head": head}))

    def forward(self, obs, actions):
        nodes, mask, global_feat = obs["nodes"], obs["mask"].float(), obs["global"]
        B, N, _ = nodes.shape
        
        # 🚀 修改 2：在入口处，直接用物理手段将假动作屏蔽为 0！
        # 告诉 Critic：“不要管假关节输出的随机数，它们全都是 0”
        masked_actions = actions * mask 
        
        # 用 masked_actions 替代原来的 actions 进行扩展拼接
        actions_expanded = masked_actions.unsqueeze(1).expand(B, N, -1)
        global_expanded = global_feat.unsqueeze(1).expand(B, N, -1)
        
        q_values = []
        for q_net in self.q_networks:
            h = q_net["backbone"](nodes, mask, global_feat)
            c = torch.cat([h, nodes, global_expanded, actions_expanded], dim=-1)
            q_nodes = q_net["head"](c).squeeze(-1)
            q = (q_nodes * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            q_values.append(q.view(-1, 1))
            
        return tuple(q_values)

    def q1_forward(self, obs, actions):
        return self.forward(obs, actions)[0]

# 5. 打包为 SB3 Policy
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