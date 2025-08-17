import torch
import torch.nn.functional as F
from torch.nn import Linear


class AttentionMechanism:
    def __init__(self, feature_dim):
        self.query_layer = Linear(feature_dim, feature_dim, bias=False).cuda()
        self.key_layer = Linear(feature_dim, feature_dim, bias=False).cuda()
        self.value_layer = Linear(feature_dim, feature_dim, bias=False).cuda()
        self.scale = torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32)).cuda()

    def compute_attention_weights(self, z_all, zs):
        """
        计算注意力权重，基于全局特征 (z_all) 和每个视图的特征 (zs)。

        参数:
        - z_all: 全局特征张量，形状为 [batch_size, feature_dim]
        - zs: 每个视图的特征列表，其中每个特征形状为 [batch_size, feature_dim]

        返回:
        - attention_weights: 注意力权重张量，形状为 [batch_size, view_count]
        """
        # 获取批次大小（样本数量）
        batch_size = z_all.size(0)
        # 获取视图数量
        view_count = len(zs)

        # 对全局特征 z_all 应用线性变换，得到查询向量 Q，形状为 [batch_size, feature_dim]
        Q = self.query_layer(z_all.cuda())

        # 对每个视图的特征 z_v 应用线性变换，生成键向量 K，并在第 1 维堆叠，形成 [batch_size, view_count, feature_dim]
        K = torch.stack([self.key_layer(z.cuda()) for z in zs], dim=1)

        # 对每个视图的特征 z_v 应用线性变换，生成值向量 V，并在第 1 维堆叠，形成 [batch_size, view_count, feature_dim]
        V = torch.stack([self.value_layer(z.cuda()) for z in zs], dim=1)

        # 检查 Q 的形状是否正确（[batch_size, feature_dim]）
        assert Q.shape == (
            batch_size, Q.size(-1)), f"Query shape mismatch: expected {(batch_size, Q.size(-1))}, got {Q.shape}"
        # 检查 K 的形状是否正确（[batch_size, view_count, feature_dim]）
        assert K.shape == (batch_size, view_count, K.size(
            -1)), f"Key shape mismatch: expected {(batch_size, view_count, K.size(-1))}, got {K.shape}"
        # 检查 V 的形状是否正确（[batch_size, view_count, feature_dim]）
        assert V.shape == (batch_size, view_count, V.size(
            -1)), f"Value shape mismatch: expected {(batch_size, view_count, V.size(-1))}, got {V.shape}"

        # 计算点积得分，通过 `torch.einsum` 实现 Q 和 K 的点积，结果形状为 [batch_size, view_count]
        # 同时对点积结果除以缩放因子 self.scale，以稳定梯度 scores = torch.bmm(Q.unsqueeze(1), K.transpose(1, 2)).squeeze(1) / self.scale
        scores = torch.einsum('bf,bvf->bv', Q, K) / self.scale

        # 使用 softmax 函数对每个样本的视图相关性得分进行归一化，生成注意力权重，形状为 [batch_size, view_count]
        attention_weights = F.softmax(scores, dim=1)

        # 返回注意力权重
        return attention_weights
