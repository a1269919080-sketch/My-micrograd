import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt



net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

def accuracy(y_hat,y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum().item()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def plot_history(epochs, train_loss, train_acc, test_acc):
    """专门用来画图的函数"""
    plt.figure(figsize=(10, 5))  # 设置画布大小

    # 画三条线
    # 'o-' 表示带圆点的实线，label 是图例名字
    plt.plot(epochs, train_loss, 'r--', label='Train Loss')  # 红色虚线
    plt.plot(epochs, train_acc, 'g-o', label='Train Acc')  # 绿色实线带点
    plt.plot(epochs, test_acc, 'b-o', label='Test Acc')  # 蓝色实线带点

    # 设置标签和标题
    plt.xlabel('Epoch')  # 横轴名字
    plt.ylabel('Value')  # 纵轴名字
    plt.title('Training Results')  # 标题
    plt.grid(True)  # 显示网格
    plt.legend()  # 显示图例 (根据 label 自动生成)

    # 这一步在 PyCharm 里必须有，否则窗口弹不出来
    plt.show()


def train_ch3_pycharm(net, train_iter, test_iter, loss, num_epochs, updater):
    """适配 PyCharm 的训练函数"""

    # --- [修改点 1]：初始化空列表，用来记录每一轮的数据 ---
    epochs_list = []
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    print(f"开始训练，共 {num_epochs} 轮...")

    for epoch in range(num_epochs):
        # 1. 训练一轮 (复用你之前的函数)
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)

        # 2. 测试一轮 (复用你之前的函数)
        test_acc = evaluate_accuracy(net, test_iter)

        # 3. 解包数据
        train_loss, train_acc = train_metrics

        # --- [修改点 2]：把数据存进列表 ---
        epochs_list.append(epoch + 1)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        # --- [修改点 3]：在控制台打印进度（PyCharm里看这个很方便）---
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}")

    # --- [修改点 4]：训练结束后，统一画图 ---
    print("训练结束，正在画图...")
    plot_history(epochs_list, train_loss_list, train_acc_list, test_acc_list)

if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_epochs = 10
    train_ch3_pycharm(net, train_iter, test_iter, loss, num_epochs, trainer)