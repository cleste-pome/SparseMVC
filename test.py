import argparse
import torch
from itertools import chain
from network_SaMVC import Network
from utils.dataloader import MATKind
from metric import valid

if __name__ == '__main__':
    # TODO please set Dataname and Weights Path
    Dataname = f'MSRCV1'
    weights_path = f'4.models/MSRCV1/MSRCV120241228-175826.pth'

    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--dataset', default=Dataname)
    parser.add_argument("--feature_dim", default=64)
    parser.add_argument("--hide_feature_dim", default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MATKind(args.dataset, f'datasets')
    # 获取数据集中类别的数量
    class_num = dataset.num_classes
    # 获取数据集中样本的总数
    data_size = len(dataset)
    # 获取数据集中视图的数量
    view = dataset.num_views
    # 获取每个视图的维度
    dims = list(chain.from_iterable(dataset.dims.tolist()))

    model = Network(view, dims, args.feature_dim, args.hide_feature_dim, device)
    model = model.to(device)

    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint)

    model.eval()
    print(f'Dataset[{args.dataset}] loading')
    acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, class_num, pre_train=False, con_train=True)
