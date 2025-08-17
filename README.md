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
Pre-train: The Sparse Autoencoder with Adaptive Encoding (SAA)

 67%|██████▋   | 200/300 [01:16<06:25,  3.85s/it]
Early-fused Feature Clustering
+-----------+--------+--------+--------+----------+
| Feature   |    ACC |    NMI |    ARI |   Purity |
+===========+========+========+========+==========+
| View 1    | 0.3921 | 0.3081 | 0.1954 |   0.4115 |
+-----------+--------+--------+--------+----------+
| View 2    | 0.4602 | 0.4355 | 0.2839 |   0.5063 |
+-----------+--------+--------+--------+----------+
| View 3    | 0.4014 | 0.3497 | 0.2252 |   0.4427 |
+-----------+--------+--------+--------+----------+
| View 4    | 0.2392 | 0.1324 | 0.0581 |   0.2604 |
+-----------+--------+--------+--------+----------+
| z_all     | 0.5275 | 0.4908 | 0.3270 |   0.5647 |
+-----------+--------+--------+--------+----------+
Sparsity ratio(zero(missing)_value(dims)_proportion mean)[view]:[0.0004, 0.0004, 0.0026, 0.1253]
```

