import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np

# 1. 准备数据
# 模拟一个简单的线性数据：y = 2x + noise
torch.manual_seed(42)
x = torch.linspace(0, 1, 100).reshape(-1, 1)  # 输入特征
y = 2 * x + 0.1 * torch.randn_like(x)         # 标签，加上一些噪声

# 2. 定义一个简单的模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. 初始化 TensorBoard 日志记录器
writer = SummaryWriter(log_dir="runs/simple_linear_regression")

# 5. 训练模型
epochs = 100
for epoch in range(epochs):
    # 模拟训练过程
    model.train()
    predictions = model(x)
    loss = criterion(predictions, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录到 TensorBoard
    writer.add_scalar("Loss/train", loss.item(), epoch)  # 记录损失值
    writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)  # 记录学习率

    # 模拟权重和偏置的变化
    for name, param in model.named_parameters():
        writer.add_histogram(f"Parameters/{name}", param, epoch)

    # 每10个epoch记录一次预测图
    if epoch % 10 == 0:
        with torch.no_grad():
            predictions = model(x)
            writer.add_scalars("Predictions vs Ground Truth", 
                               {"Predictions": predictions.mean().item(), 
                                "Ground Truth": y.mean().item()}, epoch)

# 6. 记录网络结构
writer.add_graph(model, x)

# 7. 关闭 TensorBoard 记录器
writer.close()
