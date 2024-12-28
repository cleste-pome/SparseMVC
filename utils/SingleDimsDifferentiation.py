import os
import torch
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def plot_feature_separation_kline(tensor, name, dataname):
    # 确保文件夹存在
    output_folder = f'5.SingleDimsDifferentiation/{dataname}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 确保文件名带有 .png 后缀
    if not name.endswith(".png"):
        save_path = f'{output_folder}/{name}.png'
    else:
        save_path = name

    # 判断是否是 Tensor，确保输入为 PyTorch 的 Tensor 格式
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)

    # 如果 Tensor 在 GPU 上，先转移到 CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # 计算每个维度的统计信息
    min_vals = torch.min(tensor, dim=0).values
    max_vals = torch.max(tensor, dim=0).values
    q1_vals = torch.quantile(tensor, 0.25, dim=0)
    q3_vals = torch.quantile(tensor, 0.75, dim=0)
    median_vals = torch.median(tensor, dim=0).values

    # 将这些 Tensor 数据转换为 NumPy 数组
    min_vals = min_vals.detach().numpy()
    max_vals = max_vals.detach().numpy()
    q1_vals = q1_vals.detach().numpy()
    q3_vals = q3_vals.detach().numpy()
    median_vals = median_vals.detach().numpy()

    num_features = tensor.shape[1]

    # 动态调整图像的宽和高
    width = 12  # 固定宽度
    if num_features <= 20:
        height = 6
    elif num_features <= 50:
        height = 5
    elif num_features <= 150:
        height = 4
    elif num_features <= 300:
        height = 3.5
    else:
        height = 3  # 使图像更扁

    # 自动计算X轴标签的间隔，避免标签挤在一起
    if num_features <= 20:
        tick_interval = 1
    elif num_features <= 50:
        tick_interval = 5
    elif num_features <= 150:
        tick_interval = 10
    elif num_features <= 300:
        tick_interval = 20
    else:
        tick_interval = max(20, num_features // 20)

    # 计算每个维度的K线长度，用于动态调整颜色
    candle_lengths = max_vals - min_vals
    norm = plt.Normalize(vmin=candle_lengths.min(), vmax=candle_lengths.max())
    cmap = cm.get_cmap('coolwarm')  # 使用颜色渐变

    # 设置图像，增加一个位置给颜色条
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(right=0.85)  # 留出空间给颜色条
    cax = fig.add_axes([0.86, 0.15, 0.01, 0.7])  # 调整颜色条的宽度（0.01）;把 0.88 改成 0.86，使颜色条靠近框图

    # 绘制空心蜡烛图，颜色根据K线长度变化
    for i in tqdm(range(num_features)):
        # 选择颜色
        color = cmap(norm(candle_lengths[i]))

        # 上影线
        ax.plot([i, i], [q3_vals[i], max_vals[i]], color=color, lw=1.2)
        # 下影线
        ax.plot([i, i], [q1_vals[i], min_vals[i]], color=color, lw=1.2)

        # 绘制开盘价-收盘价空心框
        ax.add_patch(plt.Rectangle((i - 0.2, q1_vals[i]), 0.4, q3_vals[i] - q1_vals[i],
                                   fill=False, edgecolor=color, lw=1.2))

    # 设置 X 轴标记：自动选择合适的间隔，标签不旋转
    xticks = list(range(0, num_features, tick_interval))
    if num_features - 1 not in xticks:
        xticks.append(num_features - 1)

    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i + 1) for i in xticks], ha='center')  # 标签水平居中显示

    ax.set_xlabel('Feature Dimension')
    ax.set_ylabel('Value')
    ax.set_title(f'Feature Separation ({name})')

    # 添加颜色条，展示K线长度和颜色的关系
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Candle Length')  # 颜色条标签

    # 保存图片
    try:
        plt.savefig(save_path, format='png')
        print(f"Figure saved at {save_path}")
    except Exception as e:
        print(f"Error saving the figure: {e}")

    # 显示图片
    plt.show()

    # 关闭图像资源以避免占用
    plt.close(fig)


def generate_random_tensor_with_different_distributions(samples, features, device='cpu'):
    # 创建一个空的 tensor
    tensor = torch.zeros((samples, features), device=device)

    for i in range(features):
        # 为每个维度选择不同的均值和标准差，来生成不同分布的随机数
        mean = np.random.uniform(-3, 3)  # 均值在 [-3, 3] 范围内随机选择
        std = np.random.uniform(0.5, 2)  # 标准差在 [0.5, 2] 范围内随机选择

        # 使用不同的均值和标准差生成该维度的数据
        tensor[:, i] = torch.randn(samples, device=device) * std + mean

    return tensor


def feature_separation(feature_list, names, dataname):
    feature_iter = 0
    for feature in feature_list:
        name = names[feature_iter]
        if torch.is_tensor(feature):
            print(f'Tensor:{name}')
            plot_feature_separation_kline(feature, name, dataname)
            feature_iter += 1
        elif isinstance(feature, list):
            print(f'List:{name}')
            name_iter = 0
            for feature_ in feature:
                name_ = f'{name}_{name_iter}'
                plot_feature_separation_kline(feature_, name_, dataname)
                name_iter += 1
            feature_iter += 1
        else:
            print(f"The variable is neither a Tensor nor a List.")
            feature_iter += 1


if __name__ == '__main__':
    # 示例：生成一个 1000 个样本、128 个不同分布维度的 tensor
    tensor = generate_random_tensor_with_different_distributions(1000, 128, device='cuda')
    plot_feature_separation_kline(tensor, name='test', dataname='demo')  # 保存图片
