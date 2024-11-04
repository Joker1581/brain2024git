import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 定义配置类
class Config:
    def __init__(self):
        self.n_components = 0.95  # PCA 保留的方差比例
        self.batch_size = 32
        self.epochs = 20
        self.learning_rate = 1e-3
        self.num_classes = 2  # 假设有两种情感
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, num_channels=64, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128 * (4608 // 2), num_classes)  # 假设经过一次池化后时间长度减半

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        return x

# PCA 降噪函数
def pca_denoising(data, n_components=0.95):
    """
    使用 PCA 进行降噪。
    data: 形状为 (samples, channels, time_points)
    n_components: 保留的主成分数量或方差比例
    """
    samples, channels, time_points = data.shape
    pca = PCA(n_components=n_components)
    denoised_data = np.zeros_like(data)
    
    for i in range(samples):
        # 对每个样本的所有通道进行标准化
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[i, :, :].T)  # 形状为 (time_points, channels)
        
        # 应用 PCA
        transformed = pca.fit_transform(data_scaled)
        reconstructed = pca.inverse_transform(transformed)
        
        # 反标准化
        denoised = scaler.inverse_transform(reconstructed)
        denoised_data[i, :, :] = denoised.T  # 转回 (channels, time_points)
    
    return denoised_data

# 生成随机数据
def generate_random_data(samples=800, channels=64, time_points=4608, num_classes=2):
    """
    生成随机的脑波数据和标签。
    """
    data = np.random.randn(samples, channels, time_points)
    labels = np.random.randint(0, num_classes, samples)
    return data, labels

# 数据加载和预处理
def load_and_preprocess_data():
    # 生成随机数据
    data, labels = generate_random_data()

    print(f"Original data shape: {data.shape}")

    # PCA 降噪
    denoised_data = pca_denoising(data, n_components=config.n_components)
    print(f"Denoised data shape: {denoised_data.shape}")

    # 转换为 PyTorch 张量
    X = denoised_data
    y = labels

    # 将数据转置为 (samples, channels, time_points)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader

# 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

# 测试函数
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = corrects.double() / len(test_loader.dataset)
    return epoch_loss, epoch_acc.item()

# 主函数
def main():
    # 加载和预处理数据
    train_loader, test_loader = load_and_preprocess_data()

    # 初始化模型、损失函数和优化器
    model = SimpleCNN(num_channels=64, num_classes=config.num_classes).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_losses = []
    test_losses = []
    test_accuracies = []

    # 训练循环
    for epoch in range(config.epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, config.device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, config.device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # 绘制训练和测试损失
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, config.epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, config.epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.show()
    
    # 绘制测试准确率
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, config.epochs+1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
