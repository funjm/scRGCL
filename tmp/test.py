"""
Dropout 可视化演示
展示不同 dropout 概率对网络训练和神经元激活的影响
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 生成非线性数据集（月牙形）
def generate_moons(n_samples=1000, noise=0.1):
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y

# 定义带有 Dropout 的简单神经网络
class SimpleNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, dropout_p=0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(dropout_p, X_train, y_train, X_val, y_val, epochs=100):
    """训练模型并记录训练过程"""
    model = SimpleNet(dropout_p=dropout_p)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        train_acc = (outputs.argmax(1) == y_train).float().mean().item()
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_acc = (val_outputs.argmax(1) == y_val).float().mean().item()
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    
    return train_losses, val_losses, train_accs, val_accs, model

def visualize_neuron_dropout(hidden_dim=50, dropout_p=0.9):
    """可视化 Dropout 对神经元激活的影响"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # 创建输入数据
    np.random.seed(42)
    inputs = torch.randn(100, hidden_dim)
    
    # 前向传播获取激活值
    dropout_layer = nn.Dropout(p=dropout_p)
    activation = torch.relu(torch.randn(100, hidden_dim))
    
    # 调试打印：解释热力图现象
    print("\n" + "="*70)
    print("热力图现象解释")
    print("="*70)
    print(f"热力图维度: (neurons, samples) = {activation[:20].numpy().T.shape}")
    print(f"  - Y轴: 神经元索引 (0-{hidden_dim-1})")
    print(f"  - X轴: 样本索引 (0-19)")
    print("\n[分析1] Neuron 0 (Y轴最顶部) 的激活值:")
    neuron0_vals = activation[:20, 0].numpy()
    print(f"  值: {neuron0_vals}")
    print(f"  最小值: {neuron0_vals.min():.4f}, 最大值: {neuron0_vals.max():.4f}")
    print(f"  平均值: {neuron0_vals.mean():.4f}")
    print(f"  → 看起来'空白'是因为这个神经元在这些样本上激活值都很小")
    print("\n[分析2] 最后一个样本 (Sample 19, X轴最右侧) 的激活值:")
    sample19_vals = activation[19, :].numpy()
    print(f"  值: {sample19_vals}")
    print(f"  大于0的数量: {(sample19_vals > 0).sum()}/{hidden_dim}")
    print(f"  平均值: {sample19_vals.mean():.4f}, 最大值: {sample19_vals.max():.4f}")
    print(f"  → 有多处值是因为这个样本在多个神经元上激活值较高")
    print("\n[原因] 这些完全是随机数生成的自然现象！")
    print("="*70 + "\n")
    
    # 训练模式下的 Dropout 输出
    activation_train = dropout_layer(activation.clone())
    
    # 提取激活统计
    original_mean = activation.mean(dim=0).numpy()
    dropped_mean = activation_train.mean(dim=0).numpy()
    
    # 左图：神经元激活对比
    ax1 = axes[0]
    x_pos = np.arange(hidden_dim)
    width = 0.35
    ax1.bar(x_pos - width/2, original_mean, width, label='Without Dropout', alpha=0.8, color='steelblue')
    ax1.bar(x_pos + width/2, dropped_mean, width, label=f'With Dropout (p={dropout_p})', alpha=0.8, color='coral')
    ax1.set_xlabel('Neuron Index', fontsize=12)
    ax1.set_ylabel('Mean Activation', fontsize=12)
    ax1.set_title('Neuron Activation: Before vs After Dropout', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_xlim(-1, hidden_dim)
    
    # 中图：不带 Dropout 的激活热力图
    ax2 = axes[1]
    im2 = ax2.imshow(activation[:20].numpy().T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=activation.max().item())
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Neuron Index', fontsize=12)
    ax2.set_title('Activation Pattern (Without Dropout)', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Activation Value')
    
    # 右图：带 Dropout 的激活热力图
    ax3 = axes[2]
    im3 = ax3.imshow(activation_train[:20].numpy().T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=activation.max().item())
    ax3.set_xlabel('Sample Index', fontsize=12)
    ax3.set_ylabel('Neuron Index', fontsize=12)
    ax3.set_title(f'Activation Pattern (With Dropout, p={dropout_p})', fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Activation Value')
    
    plt.tight_layout()
    return fig

def visualize_dropout_effect():
    """主可视化函数：展示不同 dropout 概率的影响"""
    # 生成数据
    X, y = generate_moons(n_samples=500, noise=0.2)
    X_train, X_val = X[:400], X[400:]
    y_train, y_val = y[:400], y[400:]
    
    # 转换为 PyTorch 张量
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    
    dropout_probs = [0.0, 0.2, 0.5, 0.7]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    # 训练不同 dropout 概率的模型
    results = {}
    for p in dropout_probs:
        print(f"Training model with dropout={p}...")
        train_loss, val_loss, train_acc, val_acc, model = train_model(
            p, X_train_t, y_train_t, X_val_t, y_val_t, epochs=150
        )
        results[p] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 训练损失对比
    ax1 = fig.add_subplot(2, 2, 1)
    for i, p in enumerate(dropout_probs):
        ax1.plot(results[p]['train_loss'], label=f'Dropout={p}', color=colors[i], linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 验证损失对比
    ax2 = fig.add_subplot(2, 2, 2)
    for i, p in enumerate(dropout_probs):
        ax2.plot(results[p]['val_loss'], label=f'Dropout={p}', color=colors[i], linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 训练准确率对比
    ax3 = fig.add_subplot(2, 2, 3)
    for i, p in enumerate(dropout_probs):
        ax3.plot(results[p]['train_acc'], label=f'Dropout={p}', color=colors[i], linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Training Accuracy', fontsize=12)
    ax3.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.4, 1.05])
    
    # 4. 验证准确率对比（最重要的图）
    ax4 = fig.add_subplot(2, 2, 4)
    for i, p in enumerate(dropout_probs):
        ax4.plot(results[p]['val_acc'], label=f'Dropout={p}', color=colors[i], linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation Accuracy', fontsize=12)
    ax4.set_title('Validation Accuracy Comparison (Key Metric)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.4, 1.05])
    
    # 标注过拟合区域
    ax4.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='High Accuracy')
    ax4.text(5, 0.96, 'Good Generalization Zone', fontsize=10, color='gray')
    
    plt.tight_layout()
    return fig, results

def visualize_decision_boundary():
    """可视化不同 dropout 下的决策边界"""
    X, y = generate_moons(n_samples=500, noise=0.2)
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)
    
    dropout_probs = [0.0, 0.3, 0.5]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    for idx, p in enumerate(dropout_probs):
        model = SimpleNet(dropout_p=p)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # 训练模型
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(model(X_t), y_t)
            loss.backward()
            optimizer.step()
        
        # 预测
        model.eval()
        with torch.no_grad():
            Z = model(grid).argmax(1).numpy().reshape(xx.shape)
        
        # 绘制决策边界
        axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.6, edgecolors='k')
        axes[idx].set_xlabel('Feature 1', fontsize=11)
        axes[idx].set_ylabel('Feature 2', fontsize=11)
        axes[idx].set_title(f'Dropout = {p}', fontsize=14, fontweight='bold')
        axes[idx].set_xlim(x_min, x_max)
        axes[idx].set_ylim(y_min, y_max)
    
    plt.suptitle('Decision Boundaries with Different Dropout Rates', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def visualize_sparsity(hidden_dim=100, dropout_p=0.5):
    """可视化 Dropout 造成的稀疏性"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 不同 dropout 概率
    ps = [0.1, 0.5, 0.9]
    dropout_layers = [nn.Dropout(p=p) for p in ps]
    
    # 原始激活
    activations = torch.randn(100, hidden_dim)
    
    for idx, (dropout, p) in enumerate(zip(dropout_layers, ps)):
        activated = dropout(activations.clone())
        
        # 计算激活比例
        active_ratio = (activated != 0).float().mean().item()
        
        # 绘制激活分布
        ax = axes[idx]
        active_vals = activated[activated != 0].numpy()
        
        # 左半边：柱状图
        ax.hist(active_vals, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(x=active_vals.mean(), color='red', linestyle='--', 
                   label=f'Mean: {active_vals.mean():.2f}')
        ax.axvline(x=1.0/(1-p), color='green', linestyle=':', 
                   label=f'Theoretical: {1/(1-p):.2f}')
        ax.set_xlabel('Activation Value', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Dropout p={p}\n({active_ratio*100:.1f}% neurons active)', 
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        
        # 添加注释
        ax.text(0.95, 0.95, f'Dropped: {(1-active_ratio)*100:.1f}%', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Neuron Activation Distribution After Dropout', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def print_summary(results):
    """打印训练结果摘要"""
    print("\n" + "="*60)
    print("训练结果摘要")
    print("="*60)
    print(f"{'Dropout':<10} {'最终训练损失':<15} {'最终验证损失':<15} {'最终验证准确率':<15}")
    print("-"*60)
    
    for p, data in results.items():
        final_train_loss = data['train_loss'][-1]
        final_val_loss = data['val_loss'][-1]
        final_val_acc = data['val_acc'][-1]
        print(f"{p:<10} {final_train_loss:<15.4f} {final_val_loss:<15.4f} {final_val_acc:<15.4f}")
    
    print("="*60)
    print("\n关键发现：")
    print("- Dropout=0: 容易过拟合，验证损失上升")
    print("- Dropout=0.2-0.5: 平衡性能和泛化能力")
    print("- Dropout=0.7: 过强正则化，可能欠拟合")

if __name__ == "__main__":
    print("="*60)
    print("Dropout 可视化演示")
    print("="*60)
    
    # 1. 主对比实验
    print("\n[1/4] 训练不同 Dropout 概率的模型...")
    fig1, results = visualize_dropout_effect()
    print_summary(results)
    
    # 2. 决策边界可视化
    print("\n[2/4] 生成决策边界可视化...")
    fig2 = visualize_decision_boundary()
    
    # 3. 稀疏性可视化
    print("\n[3/4] 生成神经元稀疏性可视化...")
    fig3 = visualize_sparsity()
    
    # 4. Dropout 对单个神经元的影响
    print("\n[4/4] 生成神经元激活对比图...")
    fig4 = visualize_neuron_dropout()
    
    # 保存图表
    fig1.savefig('dropout_training_comparison.png', dpi=150, bbox_inches='tight')
    fig2.savefig('dropout_decision_boundary.png', dpi=150, bbox_inches='tight')
    fig3.savefig('dropout_sparsity.png', dpi=150, bbox_inches='tight')
    fig4.savefig('dropout_neuron_activation.png', dpi=150, bbox_inches='tight')
    
    print("\n✓ 所有可视化图表已保存为 PNG 文件")
    print("✓ 图表预览将在窗口中显示")
    
    plt.show()
    print("\n演示完成！")
