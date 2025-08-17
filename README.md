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
Con-train: SAA+CSR+CDA

Late-fused Feature Clustering
+------------+--------+--------+--------+----------+
| Feature    |    ACC |    NMI |    ARI |   Purity |
+============+========+========+========+==========+
| View 1     | 0.2584 | 0.4996 | 0.1591 |   0.2929 |
+------------+--------+--------+--------+----------+
| View 2     | 0.2342 | 0.5252 | 0.1509 |   0.2672 |
+------------+--------+--------+--------+----------+
| View 3     | 0.3584 | 0.5778 | 0.2163 |   0.3949 |
+------------+--------+--------+--------+----------+
| View 4     | 0.3555 | 0.5786 | 0.2162 |   0.3891 |
+------------+--------+--------+--------+----------+
| Global (Y) | 0.5257 | 0.7069 | 0.4001 |   0.5614 |
+------------+--------+--------+--------+----------+

Sparsity ratio(zero(missing)_value(dims)_proportion mean)[view]:[0.0001, 0.0001, 0.3415, 0.6383]
```

