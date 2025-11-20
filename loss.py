import numpy as np
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块


# 定义一个对比损失类，继承自nn.Module
class ContrastiveLoss(nn.Module):
    """
    对比损失类（Contrastive Loss）。
    用于计算两个嵌入向量之间的相似性损失，主要用于无监督学习任务。
    """

    def __init__(self, batch_size, device):
        """
        初始化方法。
        :param batch_size: 每次批处理的数据量大小
        :param device: 计算设备（如'cuda'或'cpu'）
        """
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size  # 批处理大小
        self.temperature = 1.0  # 温度参数，控制相似度缩放
        self.device = device  # 计算设备

    def forward(self, h_i, h_j, weight=None):
        """
        前向传播，计算对比损失。
        :param h_i: 输入的第一个嵌入向量
        :param h_j: 输入的第二个嵌入向量
        :param weight: 可选的损失权重
        :return: 对比损失值
        """
        N = self.batch_size  # 批处理大小

        # 计算相似度矩阵（点积相似度除以温度参数）
        similarity_matrix = torch.matmul(h_i, h_j.T) / self.temperature

        # 提取正样本对的相似度（相似度矩阵的对角线元素）
        positives = torch.diag(similarity_matrix)

        # 创建一个N*N的掩码矩阵，用于屏蔽对角线元素
        mask = torch.ones((N, N), device=self.device)  # 创建全1矩阵
        mask.fill_diagonal_(0)  # 将对角线置为0

        # 计算分子：正样本对相似度的指数
        numerator = torch.exp(positives)

        # 计算分母：掩码矩阵中保留的所有元素的指数值之和
        denominator = torch.exp(similarity_matrix) * mask

        # 逐行计算损失：-log(分子/分母之和)
        loss_partial = -torch.log(numerator / torch.sum(denominator, dim=1))

        # 计算平均损失
        loss = torch.sum(loss_partial) / N

        # 如果提供了权重参数，应用权重
        if weight is not None:
            loss = weight * loss

        return loss


def kl_divergence(rho, rho_hat):
    """
    计算KL散度，用于稀疏性约束。
    :param rho: 稀疏目标值（如0.05）
    :param rho_hat: 实际的平均激活值
    :return: KL散度值
    """
    # 限制ReLU激活值范围，防止数值计算出现log(0)或溢出问题
    rho_hat = torch.clamp(rho_hat, 0 + 1e-6, 1 - 1e-6)

    # 计算KL散度
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))


def kl_sparse_loss(hidden_layer_activation, rho, sparse_beta):
    """
    计算稀疏损失，用于正则化隐藏层激活值。
    :param hidden_layer_activation: 隐藏层的激活值
    :param rho: 稀疏目标值
    :param beta: 稀疏正则化强度
    :return: 稀疏正则项
    """
    kl_total = 0.0  # 用来累加每层的 KL 稀疏损失

    for layer in range(len(hidden_layer_activation)):

        # 计算隐藏层激活值的平均值
        rho_hat = torch.mean(hidden_layer_activation[layer], dim=0)

        # 计算KL散度损失
        # kl_loss = kl_divergence(rho, rho_hat).sum()
        kl_loss = kl_divergence(rho, rho_hat).mean()

        kl_total += kl_loss

    # 返回加权的KL散度损失
    return sparse_beta * kl_total


def ae_loss_function(mean, reconstructed_x, x, hidden_layer_activation, criterion, rho=0.05, beta=1.0):
    """
    自动编码器损失函数，结合重构误差和稀疏性约束。
    :param mean: 平均值，用于稀疏性约束系数计算
    :param reconstructed_x: 重构后的数据
    :param x: 原始输入数据
    :param hidden_layer_activation: 隐藏层激活值
    :param criterion: 重构误差的损失函数
    :param rho: 稀疏目标值
    :param beta: 稀疏正则化强度
    :return: 总损失值
    """
    # 稀疏系数阈值
    threshold = 0.01

    # 计算稀疏系数 C_spa：小于阈值的值置为0，大于阈值的部分归一化到[0, 1]
    C_spa = np.where(mean <= threshold, 0, (mean - threshold) / (1 - threshold))

    # 比例系数 C_sca，默认为1
    C_sca = 1.0

    # 计算重构误差
    reconstruction_loss = criterion(reconstructed_x, x)

    # 计算稀疏性约束损失
    kl_loss = sparse_loss(hidden_layer_activation, rho, beta)

    # 总损失 = 重构误差 + 稀疏性约束损失 * 稀疏系数 * 比例系数
    ae_loss = reconstruction_loss + C_sca * C_spa * kl_loss


    return ae_loss
