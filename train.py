import argparse
import gc
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from datetime import datetime
from itertools import chain
from loss import ContrastiveLoss, ae_loss_function
from metric import valid
from utils import Logger
from utils.SingleDimsDifferentiation import feature_separation
from utils.dataloader import MATKind
from utils.metric2csv import save_lists_to_file, find_max_weighted_sum_index, find_max_last_element_index
from utils.plot import plot_acc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "5"  # 设置OMP_NUM_THREADS环境变量
print(f'1.torch version:{torch.__version__} 2.cuda available:{torch.cuda.is_available()}')


def setup_seed(Seed):
    torch.manual_seed(Seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed_all(Seed)  # 为所有GPU设置随机种子
    np.random.seed(Seed)  # 为NumPy设置随机种子
    random.seed(Seed)  # 为Python标准库的random模块设置随机种子
    torch.backends.deterministic = True  # 确保CUDA的确定性（即每次运行结果一致）


def pretrain(Epoch):
    tot_loss = 0.  # 初始化总损失
    loss_list = []  # 用于存储每个视角的损失
    criterion = torch.nn.MSELoss()  # 定义均方误差损失函数，用于计算重建误差
    # 遍历数据集，enumerate用于获取批次索引和数据
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        # 将数据从字典中提取并按键的顺序转换为张量列表
        xs_dict2tensors = [xs[key] for key in sorted(xs.keys())]
        # 将所有视角的数据拼接在一起，形成一个大的张量，用于计算整体的重建误差
        xs2one = torch.cat(xs_dict2tensors, dim=1)
        # 将每个视角的数据移动到指定设备上（例如，GPU），以便加速计算
        for v in range(view):
            xs[v] = xs[v].to(device)
        # 清空优化器中的梯度
        optimizer.zero_grad()
        # 前向传播：通过模型计算重建后的输入、隐藏表示和其他中间结果
        xrs, zs, rs, H, xr_all, z_all, activation, means = model(xs)
        # 计算平均值
        mean_average = sum(means) / len(means)
        # TODO pre 1 全局视角
        loss_list.append(
            ae_loss_function(mean_average, xs2one.to(device), xr_all.to(device), activation[0], criterion, rho=0.05,
                             beta=1.0))
        # TODO pre 2 局部视角
        for v in range(view):
            if isinstance(activation[v + 1], int):
                # 如果activation是整数（稀疏程度较低），直接使用均方误差损失
                loss_list.append(criterion(xs[v], xrs[v]))  # 计算重建损失
            else:
                # 否则，使用自定义的自编码器损失函数，考虑稀疏正则项
                loss_list.append(
                    ae_loss_function(means[v], xs[v], xrs[v], activation[v + 1], criterion, rho=0.05, beta=1.0))
        # 汇总所有视角的损失
        loss = sum(loss_list)
        # 反向传播计算梯度
        loss.backward()
        # 使用优化器更新模型参数
        optimizer.step()
        # 累加损失，用于计算当前Epoch的平均损失
        tot_loss += loss.item()
    # 计算并打印当前轮次的平均损失
    pretrain_loss = tot_loss / len(data_loader)
    print('Pre Epochs[{}]'.format(Epoch + 1), 'Loss:{:.6f}'.format(pretrain_loss))
    # 返回当前轮次的平均损失和每个视角的权重
    return pretrain_loss


def contrastive_train(Epoch, dataset_name, Plot_SDD):
    """
    CVDA：基于对比的视图级分布对齐训练过程
    :param Epoch: 当前的训练轮次
    Plot_SDD： 我发明的维度分布蜡烛图:D文章还在写（鸽子咕咕咕）
    """
    tot_loss = 0.  # 初始化总损失
    for batch_idx, (xs, _, _) in enumerate(data_loader):  # 遍历数据集
        for v in range(view):
            xs[v] = xs[v].to(device)  # 将数据移动到指定设备（如GPU）
        optimizer.zero_grad()  # 清空梯度
        xrs, zs, rs, H, xr_all, z_all, activation, means = model(xs)  # TODO 2.前向传播，获取重建后的输入、编码特征、视角一致特征和全局特征
        loss_list = []
        # if Plot_SDD:
        #     xs_list = list(xs.values())
        #     names = ['xs', 'xrs', 'zs', 'rs', 'H', 'xr_all', 'z_all']
        #     feature_separation([xs_list, xrs, zs, rs, H, xr_all, z_all], names, dataset_name)
        # TODO w2: 每个视角权重取平均
        w2 = []
        for v in range(view):
            w_v = 1 / view
            w2.append(w_v)  # 将 w_v 放入一个列表中，然后再添加到 w 中
        w2 = torch.tensor(w2).to(device)
        criterion = torch.nn.MSELoss()  # 定义均方误差损失函数
        xs_dict2tensors = [xs[key] for key in sorted(xs.keys())]
        xs2one = torch.cat(xs_dict2tensors, dim=1)
        mean_average = sum(means) / len(means)
        loss_list.append(
            ae_loss_function(mean_average, xs2one.to(device), xr_all.to(device), activation[0], criterion, rho=0.05,
                             beta=1.0))
        for v in range(view):
            if isinstance(activation[v + 1], int):
                loss_list.append(criterion(xs[v], xrs[v]))  # 计算重建损失
            else:
                loss_list.append(
                    ae_loss_function(means[v], xs[v], xrs[v], activation[v + 1], criterion, rho=0.05, beta=1.0))
        for v in range(view):
            # 自加权对比学习损失
            loss_list.append(contrastiveloss(H, rs[v], w2[v]))  # 计算对比损失
        loss = sum(loss_list)  # 汇总所有视角的损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        tot_loss += loss.item()  # 累加损失
    con_loss = tot_loss / len(data_loader)
    print('Con Epochs[{}]'.format(Epoch + 1), 'Loss:{:.6f}'.format(con_loss))  # 输出当前轮次的平均损失
    return con_loss


if __name__ == '__main__':
    # loop in data
    folder_path = "datasets"  # TODO 数据集文件夹地址
    file_names = os.listdir(folder_path)
    data_iter = 1  # 数据集位次
    for Dataname in tqdm(file_names):
        if Dataname.endswith(".mat"):
            Dataname = Dataname[:-4]
            print(
                f'---------------------------------------{Dataname}[{data_iter}]---------------------------------------')
            parser = argparse.ArgumentParser(description='train')
            parser.add_argument('--dataset', default=Dataname)
            # 超参数
            parser.add_argument('--batch_size', default=256, type=int)
            parser.add_argument("--learning_rate", default=0.0003)
            parser.add_argument("--pre_epochs", default=300)  # 300
            parser.add_argument("--con_epochs", default=300)  # 300
            parser.add_argument("--iter", default=1)
            parser.add_argument("--feature_dim", default=64)
            parser.add_argument("--high_feature_dim", default=20)
            parser.add_argument("--seed", default=50)
            parser.add_argument("--weight_decay", default=0.0)
            # TODO 选取noise ratio比例的样本，随机(1到view-1)个视图做添加高斯噪声处理
            parser.add_argument('--noise_ratio', type=float, default=0.0)
            # TODO 选取conflict ratio比例的样本，随机选择一个视图的数据用另一个类别的样本的同视图数据替换
            parser.add_argument('--conflict_ratio', type=float, default=0.0)
            # TODO 选取missing ratio比例样本的随机(1到view-1)个视图做缺失处理
            parser.add_argument('--missing_ratio', type=float, default=0.0)
            args = parser.parse_args()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # TODO log创建
            log_path = f'1.logs'
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            data_ratio = f'{args.noise_ratio}_{args.conflict_ratio}_{args.missing_ratio}'
            logger = Logger.get_logger(__file__, Dataname, data_ratio)
            dataset = MATKind(args.dataset, folder_path)
            # 获取数据集中类别的数量
            class_num = dataset.num_classes
            # 获取数据集中样本的总数
            data_size = len(dataset)
            # 获取数据集中视图的数量
            view = dataset.num_views
            # 获取每个视图的维度
            dims = list(chain.from_iterable(dataset.dims.tolist()))
            index = np.arange(data_size)
            np.random.shuffle(index)
            # TODO batch size
            if Dataname == 'MSRCV1':
                args.batch_size = 128
            elif Dataname == 'NUSWIDEOBJ':
                args.batch_size = 256
            else:
                args.batch_size = data_size
            # 数据预处理
            dataset.postprocessing(index,
                                   addNoise=True, sigma=0.5, ratio_noise=args.noise_ratio,
                                   addConflict=True, ratio_conflict=args.conflict_ratio,
                                   addMissing=True, missing_rate=args.missing_ratio)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True)
            pth_path = f'4.models'
            if not os.path.exists(f'./{pth_path}'):
                os.makedirs(f'./{pth_path}')
            acc_l, nmi_l, pur_l, ari_l, seed_l, lr_l, loss_l = [], [], [], [], [], [], []
            T = args.iter  # 循环测试次数，用于获取更准确地评价指标（平均值和方差）
            seed = args.seed
            lr = args.learning_rate
            # 生成文件名，包含当前时间，以确保文件名唯一
            current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
            imgs_path = f'2.results_imgs/{Dataname}_{current_time}'
            for i in range(T):
                print(f"ROUND:{i + 1}[seed:{seed}][learning rate:{lr}]")
                # 确定本次循环测试的随机数种子：1.固定每次结果 2.保证不同次结果不同
                setup_seed(seed)
                seed_l.append(seed)
                lr_l.append(lr)
                # 建保存评价指标的列表
                acc_list, nmi_list, pur_list, ari_list, preloss_list, conloss_list = [], [], [], [], [], []
                # TODO 重点来了: model
                from SparseMVC import Network

                model = Network(view, dims, args.feature_dim, args.high_feature_dim, device)
                print(model)
                model = model.to(device)
                state = model.state_dict()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
                contrastiveloss = ContrastiveLoss(args.batch_size, device).to(device)
                for epoch in tqdm(range(args.pre_epochs)):
                    preloss = pretrain(epoch)  # 1.pre-train
                    preloss_list.append(preloss)
                    if (epoch + 1) % 100 == 0:  # TODO pre
                        acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, class_num,
                                                   pre_train=True,
                                                   con_train=False)
                        # 将本轮pre_epochs评价指标添加到列表中
                        acc_list.append(acc)
                        nmi_list.append(nmi)
                        pur_list.append(pur)
                        ari_list.append(ari)
                # plot_acc(imgs_path, preloss_list, Dataname, 'pretrain loss')
                if data_size >= 2500:
                    args.con_epochs = 1000
                # else:
                #     args.con_epochs = 300
                for epoch in tqdm(range(args.con_epochs)):
                    epoch = args.pre_epochs + epoch
                    plot_SDD = False
                    if epoch + 1 == args.pre_epochs + args.con_epochs:
                        plot_SDD = True
                    conloss = contrastive_train(epoch, Dataname, plot_SDD)  # 2.contrastive train
                    conloss_list.append(conloss)
                    valid_check_num = 10
                    if args.con_epochs >= 1000:
                        valid_check_num = 100
                    if (epoch + 1) % valid_check_num == 0:  # TODO con
                        acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, class_num,
                                                   pre_train=False,
                                                   con_train=True)
                        acc_list.append(acc)
                        nmi_list.append(nmi)
                        pur_list.append(pur)
                        ari_list.append(ari)
                    max_index = find_max_weighted_sum_index(acc_list, nmi_list, pur_list, ari_list,
                                                            acc_weight=0.25, nmi_weight=0.25,
                                                            pur_weight=0.25, ari_weight=0.25)
                # plot_acc(imgs_path, conloss_list, Dataname, 'con loss')
                loss_list = preloss_list + conloss_list
                # TODO 1.保存最后次最后一轮的权重文件(.pth)
                state = model.state_dict()
                current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
                pth_path_meta = f'{pth_path}/' + f'{Dataname}'
                if not os.path.exists(pth_path_meta):
                    os.makedirs(pth_path_meta)
                model_path = f'{pth_path_meta}/' + args.dataset + current_time + '.pth'
                torch.save(state, model_path)
                print(f'Model(.pth) has been saved at {model_path}')
                # TODO 最后一轮
                info = {"dataset": Dataname,
                        "iter": i + 1,
                        "Last Epoch": len(acc_list) * valid_check_num,
                        "acc": acc_list[-1],
                        "Nmi": nmi_list[-1],
                        "Purity": pur_list[-1],
                        "ari": ari_list[-1],
                        "seed": seed,
                        "learning rate": lr}
                # log save
                logger.info(str(info))
                del info
                acc_l.append(acc_list)
                nmi_l.append(nmi_list)
                pur_l.append(pur_list)
                ari_l.append(ari_list)
                loss_l.append(loss_list)
                max_index = find_max_weighted_sum_index(acc_list, nmi_list, pur_list, ari_list,
                                                        acc_weight=0.25, nmi_weight=0.25,
                                                        pur_weight=0.25, ari_weight=0.25)
                # TODO 2.最好一轮(不建议这样做，除非你有early stop的理由)
                info = {"dataset": Dataname,
                        "iter": i + 1,
                        "MAX Epoch": (max_index + 1) * valid_check_num,
                        "acc": acc_list[max_index],
                        "Nmi": nmi_list[max_index],
                        "Purity": pur_list[max_index],
                        "ari": ari_list[max_index],
                        "seed": seed,
                        "learning rate": lr}
                logger.info(str(info))
                offset1 = 100000
                seed = int(abs(seed + random.uniform(-offset1, offset1)))
                offset2 = 0.0001
                lr = abs(lr + random.uniform(-offset2, offset2))
                lr = "{:.5f}".format(lr)
                lr = float(lr)
                del info
            # TODO [一般是取平均值，但考虑到需求下面实现了取最大值:D] 找到 acc_l 中最后一个元素最大的列表元素的位次（默认训练一次，所以l_max=0）
            l_max = find_max_last_element_index(acc_l)
            acc_list, nmi_list, pur_list, ari_list, loss_list = acc_l[l_max], nmi_l[l_max], pur_l[l_max], ari_l[l_max], loss_l[l_max]
            max_index = find_max_weighted_sum_index(acc_list, nmi_list, pur_list, ari_list,
                                                    acc_weight=0.25, nmi_weight=0.25,
                                                    pur_weight=0.25, ari_weight=0.25)
            plot_acc(imgs_path, acc_list, Dataname, 'acc', valid_check_num)
            plot_acc(imgs_path, nmi_list, Dataname, 'nmi', valid_check_num)
            plot_acc(imgs_path, pur_list, Dataname, 'pur', valid_check_num)
            plot_acc(imgs_path, ari_list, Dataname, 'ari', valid_check_num)

            save_lists_to_file(acc_list, nmi_list, pur_list, ari_list, loss_list, Dataname, data_ratio, valid_check_num)
            print(f'Max metric: epoch{(max_index + 1) * valid_check_num}\n'
                  f'1.acc:{acc_list[max_index] * 100:.2f}%\n'
                  f'2.nmi:{nmi_list[max_index] * 100:.2f}%\n'
                  f'3.pur:{pur_list[max_index] * 100:.2f}%\n'
                  f'4.ari:{ari_list[max_index] * 100:.2f}%\n'
                  f'5.best seed[{seed_l[l_max]}] & learning rate[{lr_l[l_max]}] for this dataset')
            # 显式删除变量
            del dataset
            # 手动调用垃圾回收
            gc.collect()
        else:
            print(f'Non-MAT file. Please convert the dataset to multi-view one-dimensional MAT format.')
        data_iter += 1
