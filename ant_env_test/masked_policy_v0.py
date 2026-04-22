import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import MultiInputPolicy

class MaskedGraphExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        # 必须调用父类初始化
        super().__init__(observation_space, features_dim)
        
        # 我们的节点有 4 个物理特征，全局躯干有 6 个特征
        node_feat_dim = 4
        global_feat_dim = 6
        
        # 1. 局部节点感知机 (Node Embedding)
        # 负责把每一个关节的物理状态映射到高维空间
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # 2. 融合感知机 (Fusion MLP)
        # 负责把聚合后的肢体信息，和躯干的前庭觉信息融合
        self.fc = nn.Sequential(
            nn.Linear(64 + global_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # 从字典中拆解物理张量
        nodes = observations["nodes"]         # 形如: (batch_size, max_nodes, 4)
        mask = observations["mask"]           # 形如: (batch_size, max_nodes)
        global_feat = observations["global"]  # 形如: (batch_size, 6)
        
        # --- 🕸️ 图网络/集合聚合核心逻辑 ---
        
        # 1. 对所有节点进行独立编码
        node_embeddings = self.node_embed(nodes) # -> (batch_size, max_nodes, 64)
        
        # 2. 🛡️ Mask 屏蔽无效肢体
        # 将不存在的腿的特征强行归零，彻底切断它们的神经信号！
        mask_expanded = mask.unsqueeze(-1).float() # 扩展维度以匹配 embedding
        masked_embeddings = node_embeddings * mask_expanded
        
        # 3. 神经信号聚合 (Sum Pooling / Message Passing 的平替版)
        # 把所有存在的腿部信号汇总到中心躯干
        aggregated_nodes = masked_embeddings.sum(dim=1) # -> (batch_size, 64)
        
        # 4. 融合全局躯干特征
        combined_feat = torch.cat([aggregated_nodes, global_feat], dim=1) # -> (batch_size, 70)
        
        # 5. 输出给 SAC 的 Actor/Critic 决定动作
        return self.fc(combined_feat)


class MaskedGraphSACPolicy(MultiInputPolicy):
    """
    为了让 pretrain_locomotion.py 代码极其简洁，
    我们直接继承 SB3 的 MultiInputPolicy，并强制把它的眼睛换成我们的 Graph 提取器！
    """
    def __init__(self, *args, **kwargs):
        # 强行注入我们的图网络特征提取器
        kwargs["features_extractor_class"] = MaskedGraphExtractor
        kwargs["features_extractor_kwargs"] = dict(features_dim=256)
        super().__init__(*args, **kwargs)