from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
from tabulate import tabulate  # 格式化表格输出


# ===================== 工具函数部分 =====================

# 计算聚类准确率（基于匈牙利算法进行类别匹配）
def cluster_acc(y_true, y_pred):
    # 将真实标签转换为整型，确保类型一致
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size, "预测标签和真实标签大小不匹配"

    # 初始化混淆矩阵
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    # 填充混淆矩阵
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # 使用匈牙利算法进行最佳类别匹配
    row, col = linear_sum_assignment(w.max() - w)
    return w[row, col].sum() / y_pred.size


# 计算纯度（Purity）
def purity(y_true, y_pred):
    # 初始化投票标签数组
    y_voted_labels = np.zeros_like(y_true)

    # 对真实标签重新编码（确保从 0 开始）
    labels = np.unique(y_true)
    label_map = {label: idx for idx, label in enumerate(labels)}
    y_true = np.vectorize(label_map.get)(y_true)

    # 为每个聚类计算投票结果
    bins = np.arange(len(labels) + 1)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        y_voted_labels[y_pred == cluster] = hist.argmax()

    # 计算并返回纯度
    return accuracy_score(y_true, y_voted_labels)


# 计算多个聚类评价指标（ACC、NMI、ARI 和 Purity）
def evaluate(y_true, y_pred):
    return {
        "nmi": v_measure_score(y_true, y_pred),  # 归一化互信息（NMI）
        "ari": adjusted_rand_score(y_true, y_pred),  # 调整兰德指数（ARI）
        "acc": cluster_acc(y_true, y_pred),  # 聚类准确率（ACC）
        "purity": purity(y_true, y_pred),  # 聚类纯度（Purity）
    }


# 打印表格工具函数（便于直观展示聚类结果）
def print_table(data, headers, title):
    print(f"\n{title}")
    print(tabulate(data, headers=headers, tablefmt="grid", floatfmt=".4f"))


# ===================== 验证函数部分 =====================

# 验证模型聚类效果
def valid(model, device, dataset, view, data_size, class_num, pre_train=False, con_train=False):
    """
    验证模型的聚类效果。
    参数说明：
    - model: 待验证的模型
    - device: 运行设备（如 GPU 或 CPU）
    - dataset: 验证数据集
    - view: 数据的视图数量
    - data_size: 数据加载器的批量大小
    - class_num: 聚类类别数量
    - pre_train: 是否为预训练阶段
    - con_train: 是否为一致性训练阶段
    """
    # 加载测试数据
    test_loader = DataLoader(dataset, batch_size=data_size, shuffle=False)
    labels = None  # 初始化真实标签

    # ===================== 数据加载与前向传播 =====================
    for batch_idx, (xs, y, _) in enumerate(test_loader): # 遍历测试数据批次
        for v in range(view):  # 遍历每个视图的数据
            xs[v] = xs[v].to(device)  # 将每个视图的数据加载到指定设备
        labels = y.cpu().detach().numpy().squeeze()  # 提取真实标签为 NumPy 数组

        # 禁用梯度计算，进行前向传播
        with torch.no_grad():
            xrs, zs, rs, Y, _, z_all, _, _ = model(xs)

    # ===================== 预训练阶段 =====================
    if pre_train:
        zs_results = []  # 保存 zs 的聚类结果

        print("\nPre-train: The Sparse Autoencoder with Adaptive Encoding (SAA)")
        for v in range(view):
            # 对每个视图的低级特征 zs[v] 进行 k-means 聚类
            metrics = evaluate(labels, KMeans(n_clusters=class_num, n_init=100).fit_predict(zs[v].cpu().numpy()))
            zs_results.append([f"View {v + 1}", metrics["acc"], metrics["nmi"], metrics["ari"], metrics["purity"]])

        # 对拼接后的 z_all 进行 k-means 聚类
        z_all_metrics = evaluate(labels, KMeans(n_clusters=class_num, n_init=100).fit_predict(z_all.cpu().numpy()))
        zs_results.append(
            ["z_all", z_all_metrics["acc"], z_all_metrics["nmi"], z_all_metrics["ari"], z_all_metrics["purity"]])

        # 打印低级特征聚类结果表格
        print_table(zs_results, headers=["Feature", "ACC", "NMI", "ARI", "Purity"],
                    title="Early-fused Feature Clustering")
        return z_all_metrics["acc"], z_all_metrics["nmi"], z_all_metrics["purity"], z_all_metrics["ari"]

    # ===================== 一致性训练阶段 =====================
    if con_train:
        rs_results = []  # 保存 rs 的聚类结果

        print("\nCon-train: SAA+SDW+CVDA")
        for v in range(view):
            # 对每个视图的一致性特征 rs[v] 进行 k-means 聚类
            metrics = evaluate(labels, KMeans(n_clusters=class_num, n_init=100).fit_predict(rs[v].cpu().numpy()))
            rs_results.append([f"View {v + 1}", metrics["acc"], metrics["nmi"], metrics["ari"], metrics["purity"]])

        # 对全局特征 Y 进行 k-means 聚类
        global_metrics = evaluate(labels, KMeans(n_clusters=class_num, n_init=100).fit_predict(Y.cpu().numpy()))
        rs_results.append(["Global (Y)", global_metrics["acc"], global_metrics["nmi"], global_metrics["ari"],
                           global_metrics["purity"]])

        # 打印一致性特征聚类结果表格
        print_table(rs_results, headers=["Feature", "ACC", "NMI", "ARI", "Purity"],
                    title="Late-fused Feature Clustering")
        return global_metrics["acc"], global_metrics["nmi"], global_metrics["purity"], global_metrics["ari"]
