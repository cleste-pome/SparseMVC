import torch
import torch.nn as nn
from torch.nn.functional import normalize
from utils.Sample_SelfWeight import AttentionMechanism


def zero_value_proportion_statistics_once(xs):
        # 创建字典（dict）用于存储每个视图
        stats_proportions = {}

        # 遍历每个视图
        for key, view in xs.items():
            # 设置阈值，考虑数据存储
            # threshold = 10 ** -6
            # zero_value_counts = (torch.abs(view) < threshold).sum(dim=1).float()

            # TODO count zero
            threshold = 0.0
            zero_value_counts = (view == threshold).sum(dim=1).float()  # 计算每个样本中零值的数量

            # 每个样本的特征/维度数
            total_elements = view.shape[1]

            # 计算每个样本的小于阈值的比例
            proportions = zero_value_counts / total_elements

            # 计算比例的均值和方差，并保留4位小数
            mean_proportion = round(torch.mean(proportions).item(), 4)
            variance_proportion = round(torch.var(proportions, unbiased=False).item(), 4)

            # 存储结果
            stats_proportions[key] = (mean_proportion, variance_proportion)

        # 缓存计算结果
        return stats_proportions


# 编码器类
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),# 线性层，将输入维度转换为500
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, 500),# 线性层，维度不变
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, 2000),# 线性层，将维度转换为2000
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2000, feature_dim)# 线性层，将维度转换为特征维度
        )
        self.hidden_layer_activation = None  # 用于保存隐藏层激活值

    def forward(self, x):
        # 前向传播时，保存500维的隐藏层激活值用于稀疏约束
        x = nn.ReLU()(self.encoder[0](x))  
        self.hidden_layer_activation = x  # 保存激活值
        for i in range(1, len(self.encoder)):
            x = self.encoder[i](x)
        # 返回编码结果和隐藏层激活值
        return x, self.hidden_layer_activation


# 解码器类
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dropout_rate=0.0):
        super(Decoder, self).__init__()  # 调用父类的构造函数
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),  # 线性层，将特征维度转换为2000
            nn.ReLU(),  # 激活函数ReLU
            nn.Dropout(dropout_rate),
            nn.Linear(2000, 500),  # 线性层，将维度转换为500
            nn.ReLU(),  # 激活函数ReLU
            nn.Dropout(dropout_rate),
            nn.Linear(500, 500),  # 线性层，维度不变
            nn.ReLU(),  # 激活函数ReLU
            nn.Dropout(dropout_rate),
            nn.Linear(500, input_dim)  # 线性层，将维度转换为输入维度
        )

    def forward(self, x):
        return self.decoder(x)  # 前向传播函数，返回解码结果


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, device):
        super(Network, self).__init__()  # 调用父类的构造函数
        self.feature_dim = feature_dim
        self.high_feature_dim = high_feature_dim
        self.view = view  # 视角数量
        self.encoders = []  # 编码器列表
        self.decoders = []  # 解码器列表
        self.mask_predictor = []
        self.input_size_all = 0
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim, 0.2).to(device))  # 创建编码器并添加到列表
            self.decoders.append(Decoder(input_size[v], feature_dim, 0.2).to(device))  # 创建解码器并添加到列表
            self.input_size_all += input_size[v]

        self.encoders.append(Encoder(self.input_size_all, feature_dim, 0.2).to(device))
        self.decoders.append(Decoder(self.input_size_all, feature_dim, 0.2).to(device))

        self.encoders = nn.ModuleList(self.encoders)  # 将编码器列表转换为ModuleList
        self.decoders = nn.ModuleList(self.decoders)  # 将解码器列表转换为ModuleList

        # 全局特征融合层
        self.feature_fusion_module = nn.Sequential(
            nn.Linear(feature_dim, 256),  # 线性层，将所有视角的特征维度合并并转换为256
            nn.ReLU(),  # 激活函数ReLU
            nn.Linear(256, high_feature_dim)  # 线性层，将维度转换为高特征维度
        )

        # 视角一致特征学习层
        self.common_information_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim)  # 线性层，将特征维度转换为高特征维度
        )

        # 循环一致性转化器
        self.cycle_transfer_module = nn.Sequential(
            nn.Linear(feature_dim, 256),  # 线性层，将所有视角的特征维度合并并转换为256
            nn.ReLU(),  # 激活函数ReLU
            nn.Dropout(0.1),
            nn.Linear(256, high_feature_dim)  # 线性层，将维度转换为高特征维度
        )

    # TODO (待定)循环一致性转化器函数
    def cycle_transfer(self, z):
        return self.cycle_transfer_module(z)

    # 全局特征融合函数
    def feature_fusion(self, zs, Wz):
        # 调整权重矩阵的形状以便与 zs 相乘
        Wz_expanded = Wz.unsqueeze(-1)  # [batch_size, view, 1]
        # 转置 zs 以便与 Wz 匹配
        zs = torch.stack(zs)
        zs_transposed = zs.permute(1, 0, 2)  # [batch_size, view, features]
        # 计算加权和
        weighted_sum = torch.sum(zs_transposed * Wz_expanded, dim=1)  # [batch_size, features]
        return normalize(self.feature_fusion_module(weighted_sum), dim=1)  # 归一化并返回融合后的特征

    def forward(self, xs):
        proportions_stats = zero_value_proportion_statistics_once(xs)
        # 输出结果
        # for key, (mean, variance, first_n) in proportions_stats.items():
        #     print(f"{key}:")
        #     print(f"  Mean: {mean}, Variance: {variance}")
        #     print(f"  First {n} proportions: {first_n}")
        means = [mean for mean, _ in proportions_stats.values()]
        print(f'Sparsity ratio(zero(missing)_value(dims)_proportion mean)[view]:{means}')

        rs = []  # 视角一致特征列表
        xrs = []  # 重建后的输入列表
        zs = []  # 编码后的特征列表
        activation = []
        xs_dict2tensors = [xs[key] for key in sorted(xs.keys())]
        xs2one = torch.cat(xs_dict2tensors, dim=1)
        z_all, hidden_activation_all = self.encoders[self.view](xs2one)
        activation.append(hidden_activation_all)

        for v in range(self.view):
            z, hidden_activation = self.encoders[v](xs[v])  # 编码输入
            activation.append(hidden_activation)
            zs.append(z)  # 添加到编码特征列表

        attention_mechanism = AttentionMechanism(self.feature_dim)
        Wz = attention_mechanism.compute_attention_weights(z_all, zs)
        # print(f'Wz:{Wz}')

        for v in range(self.view):
            xr = self.decoders[v](zs[v])
            r = normalize(self.common_information_module(zs[v]), dim=1)
            rs.append(r)  # 添加到视角一致特征列表
            xrs.append(xr)  # 添加到重建输入列表

        xr_all = self.decoders[self.view](z_all)
        H = self.feature_fusion(zs, Wz)  # 全局特征融合

        return xrs, zs, rs, H, xr_all, z_all, activation, means  # 返回重建后的输入、编码特征、视角一致特征和全局融合特征


