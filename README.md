# PINNs for Solving Schrodinger Equation

## 运行环境

首先需要使用conda配置环境，本Baseline基于Python 3.8，PyTorch1.13.0。
相较于初赛，复赛对算力的要求更高，如果想取得较好的结果，迭代步数在100,000以上，建议使用服务器进行训练。如果使用个人电脑进行训练，显存大于2G就能初步完成训练以及测试。
下述命令如果出现类似于`http erro`或是`retry XXX`，请切换为国内源重试。

```commandline
conda create -n pinn python=3.8
conda activate pinn
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

克隆本Baseline:
```commandline
git clone git@github.com:koolo233/Baseline4SolvingRBConvection.git
cd Baseline4SolvingRBConvection
```

## 本Baseline简介

本Baseline以基础PINN为模型，基于PyTorch实现了复赛的Baseline。包含从数据采样、模型构建、训练、测试以及生成提交文件的完整流程。大家可以在本Baseline的基础上进行修改，实现自己的想法。亦或是使用本Baseline的代码作为参考，实现自己的框架代码。

对于PINN基础理论以及算法框架在此不多赘述，对这部分还有疑问的选手可以参考一并附带的参考文献。

本Baseline的代码注释比较详细，对于代码中的一些细节可以参考注释。

各文件介绍如下：
1. `main.py`：主文件，包含数据采样、模型构建、训练、测试以及生成提交文件的完整流程。
2. `README.md`：本文件，包含本Baseline的简介以及使用方法。
3. `requirements.txt`：包含本Baseline的依赖库。
4. `.gitignore`：git忽略文件
5. `t50_ra1e6_pr1_s42_train_lr.npz`：训练数据集，与Kaggle上的一致
6. `LICENSE`：开源协议

从比赛页面下载测试数据集`test.csv`，放到`Baseline4SolvingSchrodingerEquation`文件夹下。
当运行本Baseline后会生成log文件夹，该文件夹中的文件介绍如下：
1. `model.pth`：训练好的模型，可以直接用于测试。
2. `baseline_submission.csv`：测试数据集的预测结果，可以直接提交到Kaggle。
3. `loss_PINN.txt`：训练过程中的损失函数日志，可以用于绘制损失函数曲线。
4. `loss_plot.png`：损失函数曲线，可以用于查看训练过程中的损失函数变化情况。
5. `test_*.png`：逐时间切片预测结果，可以用于查看预测结果。

**注意：为了减小预测文件的大小，所有输出值应当保留4位小数输出（方法可参考mian.py test()）**

希望大家能够利用初赛掌握的PINN方法，在复赛中取得好成绩。

最后：**Just Have Fun!**

## 利用PINNs求解PDE的基本流程
基本流程如下：

| 流程                | 代码位置            |
|-------------------|-----------------|
| 参数定义              | main.py 18-28   |
| 加载低分辨率数据          | main.py 31-74   |
| 构建PDE数据集          | main.py 77-123  |
| 构建神经网络            | main.py 126-163 |
| 定义基于PDE的损失计算组件    | main.py 219-250 |
| 构建优化器等训练必要的组件     | main.py 190-194 |
| 构建并执行训练循环         | main.py 166-283 |
| 针对测试数据进行模型预测并保存结果 | main.py 286-342 |
| main流程            | main.py 345-368 |

## 运行方法以及提交指南

```commandline
# 切换到项目下
cd Baseline4SolvingRBConvection
# 训练并测试
python main.py
# 在服务器上如果需要指定显卡（示例为指定0卡，如果需要使用其他卡设置为其他数值），请使用
CUDA_VISIBLE_DEVICES=0 python main.py
# 提交文件将保存到log子文件下
# 将baseline_submission.csv文件提交到Kaggle就完成了一次提交
```

## 参考文献

-- S. Esmaeilzadeh, K. Azizzadenesheli, K. Kashinath, et al. Meshfreeflownet: A physics-constrained deep continuous space-time super-resolution framework. SC20: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE, 2020: 1-15.
