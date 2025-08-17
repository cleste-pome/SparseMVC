# SparseMVC

### I have attempted to build a training framework for multi-view learning (the clustering part has already been completed). It includes functionalities such as reading datasets (.mat), data processing (with noise, missing data, and misalignment), replaceable network construction modules, loss functions, training data saving, visualization, and various other analytical utilities. You are welcome to use and reference it (paper is on road).(｡･∀･)ﾉﾞ
## Just getting started, to be continued.

## 1. Simple and seamless data loading: Simply package the dataset in .mat format and place it in the "datasets" folder to enable one-click training for all datasets, regardless of the number of views.

## 2. Data processing

```python
# TODO 选取noise ratio比例的样本，随机(1到view-1)个视图做添加高斯噪声处理
parser.add_argument('--noise_ratio', type=float, default=0.0)
# TODO 选取conflict ratio比例的样本，随机选择一个视图的数据用另一个类别的样本的同视图数据替换
parser.add_argument('--conflict_ratio', type=float, default=0.0)
# TODO 选取missing ratio比例样本的随机(1到view-1)个视图做缺失处理
parser.add_argument('--missing_ratio', type=float, default=0.0)
# TODO 选取sparsity ratio比例维度的随机(1到dims-1)个维度做置0处理
parser.add_argument('--sparsity_ratio', type=float, default=0.0)
```

## 3. Hype parameters

```python
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--pre_epochs", default=300)  # 300
parser.add_argument("--con_epochs", default=300)  # 300
parser.add_argument("--iter", default=1)
parser.add_argument("--feature_dim", default=64)
parser.add_argument("--high_feature_dim", default=20)
parser.add_argument("--seed", default=50)
parser.add_argument("--weight_decay", default=0.0)
```

## 4. Visualization of the training process and automatic data storage: The system automatically saves 1) logs, 2) images, 3) CSV files, and 4) models. It dynamically displays the evaluation metrics and loss changes for each view during training. After training is completed, it generates line plots for the loss function and evaluation metrics.

```bash
  6%|▋         | 19/300 [02:47<11:19,  2.42s/it]Con Epochs[320] Loss:16.900719
Sparsity ratio(zero(missing)_value(dims)_proportion mean)[view]:[0.0001, 0.0001, 0.3415, 0.6383]

Con-train: SAA+CSR+CDA

  7%|▋         | 20/300 [05:23<3:46:25, 48.52s/it]
Late-fused Feature Clustering
+------------+--------+--------+--------+----------+
| Feature    |    ACC |    NMI |    ARI |   Purity |
+============+========+========+========+==========+
| View 1     | 0.2245 | 0.4951 | 0.1355 |   0.2663 |
+------------+--------+--------+--------+----------+
| View 2     | 0.2381 | 0.5391 | 0.1644 |   0.2824 |
+------------+--------+--------+--------+----------+
| View 3     | 0.4129 | 0.6090 | 0.2768 |   0.4359 |
+------------+--------+--------+--------+----------+
| View 4     | 0.3731 | 0.5912 | 0.2437 |   0.4048 |
+------------+--------+--------+--------+----------+
| Global (Y) | 0.5797 | 0.7421 | 0.4529 |   0.6036 |
+------------+--------+--------+--------+----------+
Sparsity ratio(zero(missing)_value(dims)_proportion mean)[view]:[0.0001, 0.0001, 0.3415, 0.6383]
```

