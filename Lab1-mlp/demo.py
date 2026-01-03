import numpy as np
from mlp.model import MLP
from mlp.layers import Dense, Dropout, ResidualBlock, BatchNormalization, GatedResidualBlock
from mlp.activations import ReLU, Linear
from mlp.losses import MSE
from mlp.optimizers import Adam
from mlp.datasets import BostonHousingLoader
from mlp.schedulers import CosineAnnealingLR

def main():
    # 1. 加载并预处理数据集
    print("=" * 50)
    print("1. 加载 Boston Housing 数据集...")
    data_loader = BostonHousingLoader(test_size=0.2, random_state=42)
    X_train, y_train, X_test, y_test = data_loader.load_data()
    print(f" 训练集：{X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f" 测试集：{X_test.shape[0]} 样本, {X_test.shape[1]} 特征")
    print(f" 房价范围：{y_train.min():.1f} ~ {y_train.max():.1f}（千美元）")
    print("=" * 50)
    # 2. 构建 MLP 模型
    print("\n2. 构建 MLP 模型...")
    model = MLP()
    # 网络结构：输入层(13) → 隐藏层(128) → BatchNormalization → ReLU → [门控残差块x3] → 输出层(1, Linear)
    # 使用支持跨层连接的门控残差块和批量归一化构建深层网络
    # 输入层到第一个门控残差块
    model.add_layer(Dense(13, 128, weight_initializer='he'))  # 增加隐藏层大小到256
    model.add_layer(BatchNormalization())
    model.add_layer(ReLU())
    # 门控残差块1：支持跨层连接到输入层(13)
    model.add_layer(GatedResidualBlock(128, 256, 128, cross_connections=[13], weight_initializer='he'))
    # 门控残差块2：支持跨层连接到输入层(13)和第一个残差块输出(128)
    model.add_layer(GatedResidualBlock(128, 256, 128, cross_connections=[13, 128], weight_initializer='he'))
    # 门控残差块3：支持跨层连接到输入层(13)、第一个和第二个残差块输出(128)
    model.add_layer(GatedResidualBlock(128, 256, 128, cross_connections=[13, 128, 128], weight_initializer='he'))
    # 添加Dropout层（只在残差块之后添加一次）
    model.add_layer(Dropout(rate=0.4))
    # 输出层
    model.add_layer(Dense(128, 1, weight_initializer='he'))
    model.add_layer(Linear())
    
    # 设置损失函数和优化器
    model.set_loss(MSE())
    model.set_optimizer(Adam(learning_rate=0.0001))  # 进一步降低学习率以适应更大的模型
    print(" 模型结构：13 → 256 → BatchNormalization → ReLU → [残差块x4] → 1(Linear)")
    print(" 残差块：128 → BatchNormalization → ReLU → 256 → BatchNormalization → 128 + 跳跃连接")
    print(" 损失函数：MSE")
    print(" 优化器：Adam (lr=0.0001)")
    print(" 正则化：Dropout(0.4) + L2(0.001)")
    print("=" * 50)
    # 3. 训练模型
    print("\n3. 开始训练模型...")
    # 创建余弦退火学习率调度器
    scheduler = CosineAnnealingLR(initial_lr=0.001, T_max=500, eta_min=0.0001)
    history = model.train(
        X_train=X_train,
        y_train=y_train,
        epochs=500,
        batch_size=32,
        validation_data=(X_test, y_test),
        early_stopping=True,
        patience=100,
        scheduler=scheduler
    )
    print("=" * 50)
    # 4. 评估模型性能
    print("\n4. 评估模型性能...")
    test_loss, metrics = model.evaluate(X_test, y_test)
    print(f" 测试集 MSE 损失：{test_loss:.4f}")
    print(f" 平均绝对误差(MAE)：{metrics['mae']:.2f}（千美元）")
    print(f" R²评分：{metrics['r2']:.4f}（越接近 1 越好）")
    print("=" * 50)
    # 5. 可视化结果
    print("\n5. 可视化训练结果...")
    model.plot_history(save_path='results/training_curve.png')
    model.plot_predictions(X_test, y_test, save_path='results/predictions.png')
    print(" 训练曲线已保存至：results/training_curve.png")
    print(" 预测结果图已保存至：results/predictions.png")
    print("=" * 50)
    # 6. 示例预测
    print("\n6. 示例预测结果...")
    sample_indices = [0, 10, 20, 30, 40]
    X_sample = X_test[sample_indices]
    y_true_sample = y_test[sample_indices]
    y_pred_sample = model.predict(X_sample)
    for i, (true, pred) in enumerate(zip(y_true_sample, y_pred_sample)):
        error = abs(true - pred)
        print(f" 样本{i+1}：真实房价={true:.2f}, 预测房价 ={pred:.2f}, 误差={error:.2f}（千美元）")
    print("=" * 50)
if __name__ == "__main__":
    main()