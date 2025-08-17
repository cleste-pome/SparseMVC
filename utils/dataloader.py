import random
import numpy as np
import scipy.io

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    def __init__(self, data_name, data_X, data_Y):
        super(MultiViewDataset, self).__init__()
        self.data_name = data_name

        self.X = dict()
        self.num_views = data_X.shape[0]
        for v in range(self.num_views):
            self.X[v] = self.normalize(data_X[v])

        self.Y = data_Y
        self.Y = np.squeeze(self.Y)
        if np.min(self.Y) == 1:
            self.Y = self.Y - 1
        self.Y = self.Y.astype(dtype=np.int64)
        self.num_classes = len(np.unique(self.Y))
        self.dims = self.get_dims()

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.Y[index]
        return data, target, index

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])
        return np.array(dims)

    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def postprocessing(self, index,
                       addNoise=False, sigma=0, ratio_noise=0.5,
                       addConflict=False, ratio_conflict=0.5,
                       addMissing=False, missing_rate=0.5):
        if addNoise:
            self.addNoise(index, ratio_noise, sigma=sigma) # 'sigma': 噪声的标准差; ratio_noise: 决定对多少比例的数据添加噪声
        if addConflict:
            self.addConflict(index, ratio_conflict)
        if addMissing:
            self.addMissing(index, missing_rate)
        pass

    def addNoise(self, index, ratio, sigma):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            elements = list(range(self.num_views))  # 生成一个包含0到num_views-1的列表
            random.seed()  # 确保每次运行时生成不同的随机数
            length = random.randint(1, self.num_views)  # views数量为随机选取的该列表的子集长度
            views = random.sample(elements, length)  # 从该列表中随机选取views个不重复的元素
            for v in views:
                self.X[v][i] = np.random.normal(self.X[v][i], sigma)
        print(f'1. Add Noise completed: {ratio}]')
        pass

    def addConflict(self, index, ratio):
        # 初始化一个字典来记录每个类别的某个代表性数据的视图值
        records = dict()
        # 遍历每个类别
        for c in range(self.num_classes):
            # 找到类别为c的第一个数据的索引
            i = np.where(self.Y == c)[0][0]
            # 初始化一个临时字典来存储当前类别的数据的各视图值
            temp = dict()
            # 遍历所有视图
            for v in range(self.num_views):
                # 记录当前视图下，当前类别的第一个数据的值
                temp[v] = self.X[v][i]
            # 将当前类别的数据视图值存储到records字典
            records[c] = temp
        # 随机选择一部分数据索引用于添加冲突，选择的数量由比例ratio和索引总数决定
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        # 对每一个被选中添加冲突的数据索引
        for i in selects:
            # 随机选择一个视图
            v = np.random.randint(self.num_views)
            # 修改当前选择的数据索引i的视图v的值，将其设置为当前数据的类别+1后的类别对应的视图值
            # 这里使用模运算保证类别编号是循环的（即如果当前类别是最后一个类别，+1后变成第一个类别）
            self.X[v][i] = records[(self.Y[i] + 1) % self.num_classes][v]
        print(f'2. Add Conflict completed: {ratio}]')
        pass

    def addMissing(self, index, ratio):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            # 从视图的总数中随机选择一部分视图
            elements = list(range(self.num_views))  # 生成一个包含0到self.num_views-1的列表
            random.seed()  # 确保每次运行时生成不同的随机数
            length = random.randint(1, self.num_views)  # views数量为随机选取的该列表的子集长度
            views = random.sample(elements, length)  # 从该列表中随机选取views个不重复的元素
            for v in views:
                self.X[v][i] = 0
        print(f'3. Add Missing completed[ratio: {ratio}]')
        pass


def MATKind(dataset_name, path):
    data_path = f"{path}/{dataset_name}.mat"
    data = scipy.io.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']
    return MultiViewDataset(f"{dataset_name}", data_X, data_Y)
