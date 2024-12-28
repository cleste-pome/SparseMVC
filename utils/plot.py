# 导入所需的模块
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

"""
# 用例
plot_acc(acc_list, dataset.data_name, 'acc')
plot_acc(nmi_list, dataset.data_name, 'nmi')
"""


# 定义绘制准确率曲线的函数，参数acc_list为各轮训练的准确率列表
def plot_acc(imgs_path, acc_list, dataset_name, name, Valid_check_num=1):
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)

    # 获取总的训练轮数
    epochs = len(acc_list)
    # 设置绘图的大小
    plt.figure(figsize=(12, 6))
    # 绘制准确率曲线，设置线型、点标记、线宽等
    plt.plot(range(1, epochs + 1), acc_list, marker='o', linestyle='-', linewidth=2, markersize=6)

    # 设置x轴和y轴的标签及其字体大小
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(f'{name}', fontsize=14)
    # 设置图表的标题及其字体大小
    plt.title(f'{dataset_name}[{name}]', fontsize=16)

    # 计算最大准确率及其对应的轮数
    max_acc = max(acc_list)
    max_epoch = acc_list.index(max_acc) + 1
    # 获取最后一轮的准确率
    last_acc = acc_list[-1]

    # 绘制表示最大准确率的水平线
    plt.axhline(y=max_acc, color='gray', linestyle='--', linewidth=0.5)
    # 在图表上标注最大准确率及其对应的轮数
    plt.text(epochs, max_acc, f'Max Acc: {max_acc * 100:.2f}% at Epoch {max_epoch * Valid_check_num}', ha='right',
             va='bottom',
             fontsize=10)
    # 在图表上标注最后一轮的准确率
    plt.text(1, 0, f'Last {name}: {last_acc * 100:.2f}%', ha='right', va='bottom', fontsize=10,
             transform=plt.gca().transAxes)

    # 设置x轴的刻度，如果训练轮数多于100轮，减少显示的刻度以避免拥挤
    if epochs > 100:
        step = epochs // 10
        plt.xticks(range(1, epochs + 1, step))
    else:
        plt.xticks(range(1, epochs + 1))

    # 设置y轴的刻度
    plt.yticks(np.arange(min(acc_list), max(acc_list) + 0.05, step=0.05))
    # 设置仅在y轴方向显示网格线
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    # 自动调整子图参数，确保图表的元素不会重叠
    plt.tight_layout()

    # TODO 文件名
    filename = f'{imgs_path}/{dataset_name}_ep{epochs}_{name}.png'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # 保存图表为PNG文件，指定分辨率为300dpi
    plt.savefig(filename, dpi=300)
    # 显示图表
    plt.show()
    # 打印保存的图表文件名
    print(f'Plot saved as {filename}')