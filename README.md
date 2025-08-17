# SparseMVC

I have attempted to build a training framework for multi-view learning (the clustering part has already been completed). It includes functionalities such as reading datasets (.mat), data processing (with noise, missing data, and misalignment), replaceable network construction modules, loss functions, training data saving, visualization, and various other analytical utilities. You are welcome to use and reference it (paper is on road).(｡･∀･)ﾉﾞ

Just getting started, to be continued.

## 1. Workflow of SparseMVC

The framework of SparseMVC.

## 2. Requirements

- python==3.8.15

- pytorch==1.12.0

- numpy==1.21.6

- scikit-learn==1.0

- scipy==1.10.1

## 3. Multi-View Clustering Code Framework

### 3.1 Simple and seamless data loading

Simply package the dataset in .mat format and place it in the "datasets" folder to enable one-click training for all datasets, regardless of the number of views.

### 3.2 Data processing

```python
# TODO 中文：选取noise ratio比例的样本，随机(1到view-1)个视图做添加高斯噪声处理 English：Select samples with a noise ratio, and randomly apply Gaussian noise processing to (1 to view-1) views
parser.add_argument('--noise_ratio', type=float, default=0.0)
# TODO 中文：选取conflict ratio比例的样本，随机选择一个视图的数据用另一个类别的样本的同视图数据替换 English：Select samples with a conflict ratio, and randomly choose data from one view to replace it with the same view data from a sample of another category
parser.add_argument('--conflict_ratio', type=float, default=0.0)
# TODO 中文：选取missing ratio比例样本的随机(1到view-1)个视图做缺失处理 English：Select samples with a missing ratio, and randomly apply missing data processing to (1 to view-1) views
parser.add_argument('--missing_ratio', type=float, default=0.0)
# TODO 中文：选取sparsity ratio比例维度的随机(1到dims-1)个维度做置0处理 English：Select dimensions with a sparsity ratio, and randomly apply zeroing to (1 to dims-1) dimensions
parser.add_argument('--sparsity_ratio', type=float, default=0.0)
```

### 3.3 Hype parameters

```python
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--pre_epochs", default=300)  
parser.add_argument("--con_epochs", default=300)  
parser.add_argument("--iter", default=1)
parser.add_argument("--feature_dim", default=64)
parser.add_argument("--high_feature_dim", default=20)
parser.add_argument("--seed", default=50)
parser.add_argument("--weight_decay", default=0.0)
```

### 3.4 Visualization of the training process and automatic data storage

The system automatically saves 

1. logs: training information (Facilitate manual review);
2. images: the change curve of evaluation metrics during the training process;
3. CSV files: metrics information (Facilitate subsequent plotting and reading);
4. models: network weight file in pt format. 

It dynamically displays the evaluation metrics and loss changes for each view during training. After training is completed, it generates line plots for the loss function and evaluation metrics.

```bash
 30%|██▉       | 89/300 [21:44<08:51,  2.52s/it]Con Epochs[389] Loss:16.679317
Sparsity ratio(zero(missing)_value(dims)_proportion mean)[view]:[0.0001, 0.0001, 0.3415, 0.6383]

 30%|███       | 90/300 [24:21<2:51:13, 48.92s/it]Con Epochs[390] Loss:16.677748
Sparsity ratio(zero(missing)_value(dims)_proportion mean)[view]:[0.0001, 0.0001, 0.3415, 0.6383]

Con-train: SAA+CSR+CDA

Late-fused Feature Clustering
+------------+--------+--------+--------+----------+
| Feature    |    ACC |    NMI |    ARI |   Purity |
+============+========+========+========+==========+
| View 1     | 0.4063 | 0.6290 | 0.2830 |   0.4392 |
+------------+--------+--------+--------+----------+
| View 2     | 0.3730 | 0.6155 | 0.2562 |   0.4012 |
+------------+--------+--------+--------+----------+
| View 3     | 0.5052 | 0.6727 | 0.3628 |   0.5235 |
+------------+--------+--------+--------+----------+
| View 4     | 0.4739 | 0.6509 | 0.3437 |   0.4997 |
+------------+--------+--------+--------+----------+
| Global (Y) | 0.7216 | 0.8175 | 0.5998 |   0.7358 |
+------------+--------+--------+--------+----------+
```

## 4. Acknowledgments

Our proposed SparseMVC draws inspiration from the works of [SCMVC](https://github.com/SongwuJob/SCMVC),  [SDMVC](https://github.com/SubmissionsIn/SDMVC) and [CPSPAN](https://github.com/jinjiaqi1998/CPSPAN). We would like to thank the authors for their valuable contributions.

## 5.Citation

If you use our code framework or get inspired by our work (hopefully as a positive example, but if it’s a negative example, that's fine too(*≧ω≦)), please cite our paper! 

```latex

```

