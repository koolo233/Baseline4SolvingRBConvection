"""
本文件为2023流光杯复赛Baseline

本文件包含完整的采样、模型构建、训练以及输出预测等功能，用于参赛选手参考实现湍流超分。

File Name: main.py
Author: Zijiang Yang
Created Date: 2023-12-06
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
supervised_data_bs = 4096  # Batch size for supervised data
pde_data_bs = 8192  # Batch size for PDE
MAX_EPOCHS = 1000  # 最大训练epoch
Rayleigh = 1e6  # Rayleigh number
Prandtl = 1  # Prandtl number
R = (Rayleigh / Prandtl)**(-1/2)  # R = np.sqrt(Prandtl / Rayleigh)
P = (Rayleigh * Prandtl)**(-1/2)  # P = 1 / np.sqrt(Rayleigh * Prandtl)
if_plot_gt = False  # 是否绘制gt数据
os.makedirs('./log',exist_ok=True)  # log输出文件夹


def load_lr_data():
    # 加载低分辨率数据
    lr_data = np.load('./t50_ra1e6_pr1_s42_train_lr.npz')
    sim_times = list(lr_data.keys())
    res_x, res_y = lr_data[sim_times[0]].shape[1:]
    res_space = res_x * res_y

    n_total = len(sim_times) * res_space
    lr_data_total = np.zeros((n_total, 7))
    for i, sim_time in enumerate(sim_times):
        sim_time_array = np.ones((res_space, 1)) * eval(sim_time)
        lr_data_total[i * res_space:(i + 1) * res_space, :] = np.concatenate([
            sim_time_array,
            lr_data[sim_time].transpose(1, 2, 0).reshape(res_space, 6)
        ], axis=1)

    # plot
    if if_plot_gt:
        plt.figure(figsize=(15, 8))
        clims = [(-.5, .5), (-.5, .5), (-.5, .5), (-.15, .15)]
        os.makedirs('./log/gt', exist_ok=True)
        print('plot gt data...')
        for i, sim_time in enumerate(sim_times):
            sim_time = eval(sim_time)
            input_slice = lr_data_total[i * res_space:(i + 1) * res_space, :3].reshape(res_x, res_y, 3)
            x_arange = input_slice[:, 0, 1].reshape(1, -1)
            y_arange = input_slice[0, :, 2].reshape(-1, 1)
            pred_slice = lr_data_total[i * res_space:(i + 1) * res_space, 3:].reshape(res_x, res_y, 4)

            plt.clf()
            key_list = ['u', 'w', 'T', 'P']
            for t in range(4):
                plt.subplot(2, 2, t+1)
                plt.title(f"{key_list[t]}: time:{sim_time:.2f}s")
                plt.pcolormesh(x_arange, y_arange, pred_slice[:, :, t].T, cmap="seismic", vmin=clims[0][0],
                               vmax=clims[0][1])
                plt.xlabel('x')
                plt.ylabel('y')
                plt.colorbar(orientation='horizontal')

            plt.savefig(f'log/gt/gt_{i}.png')
        plt.close()

    return lr_data_total


def create_data(plot_fig=True):
    """
    创建用于监督PDE、Boundary以及Initial的数据集
    :return: 创建的数据集
    """

    # 根据待求解方程确定x范围和t范围
    Lx = 4.0
    Ly = 1.0
    x_lower = 0        # x最小值
    x_upper = Lx       # x最大值
    y_lower = -Ly / 2  # z最小值
    y_upper = Ly / 2   # z最大值
    t_lower = 0        # t最小值
    t_upper = 50.      # t最大值

    # ----------
    # PDE
    # ----------
    # 创建PDE采样点数据集
    # 需要在区间内任意采样
    # 总样本点数为2000
    x_collocation = np.random.uniform(low=x_lower, high=x_upper, size=(pde_data_bs, 1))
    y_collocation = np.random.uniform(low=y_lower, high=y_upper, size=(pde_data_bs, 1))
    t_collocation = np.random.uniform(low=t_lower, high=t_upper, size=(pde_data_bs, 1))
    x_collocation_tensor = torch.from_numpy(x_collocation).float()
    y_collocation_tensor = torch.from_numpy(y_collocation).float()
    t_collocation_tensor = torch.from_numpy(t_collocation).float()
    pde_data = torch.cat([t_collocation_tensor, x_collocation_tensor, y_collocation_tensor], 1).requires_grad_(True).to(device)

    # ----------
    # 绘制采样结果
    # ----------
    # plot 3D scatter
    if plot_fig:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_collocation, y_collocation, t_collocation, marker='.', c='r')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('t')
        plt.title('Collocation Points')
        plt.savefig(f'./log/Collocation Points.png')
        plt.clf()
        plt.close()

    return pde_data


class PINN(nn.Module):
    """
    PINN 模型定义
    """
    def __init__(self, num_layers, num_neurons, input_dim=3, output_dim=4):
        """
        模型初始化
        :param num_layers: 总层数
        :param num_neurons: 每一层神经元数量
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        """
        super(PINN, self).__init__()

        # 输入层
        self.input_layer = nn.Linear(input_dim, num_neurons[0])

        # 隐藏层
        # 每一层由线性层和非线性激活函数构成
        layers = []
        for i in range(1, num_layers):
            layers.append(nn.Linear(num_neurons[i-1], num_neurons[i]))
            layers.append(nn.Tanh())
        self.hidden_layers = nn.Sequential(*layers)

        # 输出层
        self.output_layer = nn.Linear(num_neurons[-1], output_dim)

    def forward(self, x):
        """
        正向forward函数
        :param x: 输入数据
        :return: 结果
        """
        out = torch.tanh(self.input_layer(x))
        out = self.hidden_layers(out)
        out_final = self.output_layer(out)
        return out_final


def train(model, supervised_data):
    """
    训练主函数
    本函数中将定义损失函数、优化器以及训练过程
    :param model: PINN模型
    :param supervised_data: 监督数据
    :return: None
    """

    def gradients(_output, _input_tensor):
        """
        梯度计算
        :param _output: 输出tensor
        :param _input_tensor: 输入tensor
        :return: 输出对输入的梯度计算结果
        """
        _gradients = torch.autograd.grad(
            outputs=_output,
            inputs=_input_tensor,
            grad_outputs=torch.ones_like(_output),
            create_graph=True
        )[0]
        return _gradients

    losses = []             # 损失记录器
    pde_losses = []         # PDE损失记录器
    supervised_losses = []  # 监督损失记录器

    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 初始化优化器

    supervised_data_n = supervised_data.shape[0]
    supervised_data_idx = np.arange(supervised_data_n)
    for epoch in range(1, MAX_EPOCHS + 1):
        pde_data = create_data(plot_fig=False)  # 创建PDE监督样本
        np.random.shuffle(supervised_data_idx)  # 打乱监督数据
        supervised_data = supervised_data[supervised_data_idx, :]
        for step in range(1, supervised_data_n // supervised_data_bs + 1):
            model.train()
            optimizer.zero_grad()

            # -------------
            # 监督损失
            # -------------
            supervised_data_batch = supervised_data[(step-1)*supervised_data_bs:step*supervised_data_bs, :]
            supervised_data_batch_input = torch.from_numpy(supervised_data_batch[:, 0:3]).float().to(device)
            supervised_data_batch_output = torch.from_numpy(supervised_data_batch[:, 3:7]).float().to(device)

            pred = model(supervised_data_batch_input)
            supervised_loss = torch.mean((pred - supervised_data_batch_output) ** 2)

            # -------------
            # PDE损失
            # -------------
            output = model(pde_data)  # PDE监督样本结果
            pred_u, pred_w, pred_t, pred_p = output[:, 0:1], output[:, 1:2], output[:, 2:3], output[:, 3:4]

            # first order
            du_dtxy = gradients(pred_u, pde_data)
            dw_dtxy = gradients(pred_w, pde_data)
            dt_dtxy = gradients(pred_t, pde_data)
            dp_dtxy = gradients(pred_p, pde_data)

            # second order
            du_dxx = gradients(du_dtxy[:, 1:2], pde_data)[:, 1:2]
            dw_dxx = gradients(dw_dtxy[:, 1:2], pde_data)[:, 1:2]
            dt_dxx = gradients(dt_dtxy[:, 1:2], pde_data)[:, 1:2]
            du_dyy = gradients(du_dtxy[:, 2:3], pde_data)[:, 2:3]
            dw_dyy = gradients(dw_dtxy[:, 2:3], pde_data)[:, 2:3]
            dt_dyy = gradients(dt_dtxy[:, 2:3], pde_data)[:, 2:3]

            # x axis momentum
            pde_x_momentum = du_dtxy[:, 0:1] + pred_u * du_dtxy[:, 1:2] + pred_w * du_dtxy[:, 2:3] + \
                             dp_dtxy[:, 1:2] - R * (du_dxx + du_dyy)
            # z axis momentum
            pde_z_momentum = dw_dtxy[:, 0:1] + pred_u * dw_dtxy[:, 1:2] + pred_w * dw_dtxy[:, 2:3] + \
                             dp_dtxy[:, 2:3] - R * (dw_dxx + dw_dyy) - pred_t
            # energy
            pde_energy = dt_dtxy[:, 0:1] + pred_u * dt_dtxy[:, 1:2] + pred_w * dt_dtxy[:, 2:3] - \
                            P * (dt_dxx + dt_dyy)
            # continuity
            pde_continuity = du_dtxy[:, 1:2] + dw_dtxy[:, 2:3]

            # 计算总PDE损失
            pde_loss = torch.mean(pde_x_momentum ** 2) + torch.mean(pde_z_momentum ** 2) + \
                          torch.mean(pde_energy ** 2) + torch.mean(pde_continuity ** 2)

            # 误差累加
            loss = supervised_loss + pde_loss

            loss.backward()  # 反向传播
            optimizer.step()  # 参数优化

            # 记录损失值
            losses.append(loss.item())
            pde_losses.append(pde_loss.item())
            supervised_losses.append(supervised_loss.item())

            # 输出字符串
            print_str = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4e}, supervised_loss: {:.4e}, pde_loss:{:.4e}'.format(
                epoch, MAX_EPOCHS, step, supervised_data_n // supervised_data_bs, loss.item(), supervised_loss.item(), pde_loss.item()
            )
            # 输出到log文件
            with open('log/loss_PINN.txt', 'a') as f:
                f.write(print_str + '\n')
            # 输出到cmd
            print(print_str)

    # 保存模型
    torch.save(model.state_dict(), f"./log/model.pth")

    # 绘制损失图像
    plt.plot(range(1, len(losses) + 1), losses, label='Loss')
    plt.plot(range(1, len(pde_losses) + 1), pde_losses, label='pde_loss')
    plt.plot(range(1, len(supervised_losses) + 1), supervised_losses, label='supervised_loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'log/loss_plot.png')


def testing(test):

    model.eval()
    n_test_point = len(test['t'])
    pred = np.zeros((n_test_point, 4))
    batch_size = 10000
    for i in range(n_test_point // batch_size + 1):
        # 创建test输入数据
        test_input_t = torch.from_numpy(np.array(test['t'][i*batch_size:(i+1)*batch_size])).float().to(device)
        test_input_x = torch.from_numpy(np.array(test['x'][i*batch_size:(i+1)*batch_size])).float().to(device)
        test_input_y = torch.from_numpy(np.array(test['y'][i * batch_size:(i + 1) * batch_size])).float().to(device)
        test_input_tensor = torch.stack([test_input_t, test_input_x, test_input_y], dim=1)

        # 模型推理
        print(f"Testing: {i * batch_size}/{n_test_point}")
        with torch.no_grad():
            pred_batch = model(test_input_tensor.to(device)).detach()
        pred[i*batch_size:(i+1)*batch_size, :] = pred_batch.cpu().numpy()

    # 输出到csv
    df = pd.DataFrame()  # 创建DataFrame
    df["id"] = range(test['t'].shape[0])  # 创建id列
    df["t"] = test['t']  # 创建t列
    df["x"] = test['x']  # 创建x列
    df["y"] = test['y']  # 创建z列
    df["u"] = pred[:, 0]  # 创建u列
    df["w"] = pred[:, 1]  # 创建w列
    df["T"] = pred[:, 2]  # 创建t列
    df["P"] = pred[:, 3]  # 创建p列
    df["u"] = df["u"].apply(lambda x: round(x, 4))
    df["w"] = df["w"].apply(lambda x: round(x, 4))
    df["T"] = df["T"].apply(lambda x: round(x, 4))
    df["P"] = df["P"].apply(lambda x: round(x, 4))
    df.to_csv(f"log/baseline_submission.csv", index=False)

    # plot
    x_arange = np.array(test['x'][0:512 * 128]).reshape(512, 128)[:, 0].reshape(1, -1)
    y_arange = np.array(test['y'][0:512 * 128]).reshape(512, 128)[0, :].reshape(-1, 1)
    plt.figure(figsize=(15, 8))
    clims = [(-.5, .5), (-.5, .5), (-.5, .5), (-.15, .15)]

    total_n = int(len(test['t']) / 512 / 128)
    for i in range(total_n):
        pred_slice = pred[i*512*128:(i+1)*512*128, :].reshape(512, 128, 4)

        plt.clf()
        key_list = ['u', 'w', 'T', 'P']
        for t in range(4):
            plt.subplot(2, 2, t + 1)
            plt.title(f"{key_list[t]}: time:{test['t'][i*512*128]:.2f}s")
            plt.pcolormesh(x_arange, y_arange, pred_slice[:, :, t].T, cmap="seismic", vmin=clims[0][0],
                           vmax=clims[0][1])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.colorbar(orientation='horizontal')
        plt.savefig(f'log/test_{i}.png')
    plt.close()


if __name__ == '__main__':
    # 监督数据
    loaded_supervised_data = load_lr_data()
    print("Init data done...")

    # ----------
    # 创建模型
    # ----------
    model = PINN(num_layers=3, num_neurons=[64, 64, 64]).to(device)
    print("Init model done...")
    print(model)

    # ----------
    # 训练主循环
    # ----------
    train(model, loaded_supervised_data)
    print("Training done...")

    # ----------
    # 测试主循环
    # ----------
    testing(pd.read_csv('test.csv'))  # prediction
    print("Testing Done...")
    print("prediction file is saved as log/baseline_submission.csv")
