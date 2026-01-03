import itertools
from mlp.model import MLP
from mlp.layers import Dense
from mlp.activations import ReLU, Linear
from mlp.losses import MSE
from mlp.optimizers import Adam, SGD
from mlp.datasets import BostonHousingLoader
def grid_search():
    # 定义超参数网格
    param_grid = {
    'hidden_sizes': [(64, 32), (128, 64), (32, 16)],
    'learning_rate': [0.001, 0.0005],
    'batch_size': [32, 64],
    'optimizer': [Adam, lambda lr: SGD(lr=lr, momentum=0.9)]
    }
    # 加载数据
    data_loader = BostonHousingLoader(random_state=42)
    X_train, y_train, X_test, y_test = data_loader.load_data()
    best_r2 = 0
    best_params = None
    best_model = None
    # 遍历所有超参数组合
    for params in itertools.product(*param_grid.values()):
        hidden_sizes, lr, batch_size, optimizer_cls = params
        optimizer = optimizer_cls(lr) if callable(optimizer_cls) else optimizer_cls(learning_rate=lr)
    # 构建模型
    model = MLP()
    model.add_layer(Dense(13, hidden_sizes[0]))
    model.add_layer(ReLU())
    model.add_layer(Dense(hidden_sizes[0], hidden_sizes[1]))
    model.add_layer(ReLU())
    model.add_layer(Dense(hidden_sizes[1], 1))
    model.add_layer(Linear())
    model.set_loss(MSE())
    model.set_optimizer(optimizer)
    # 训练模型
    print(f"训练参数：hidden_sizes={hidden_sizes}, lr={lr}, batch_size={batch_size}, optimizer={optimizer.__class__.__name__}")
    model.train(X_train, y_train, epochs=80,
    batch_size=batch_size, validation_data=(X_test, y_test))
    # 评估模型
    _, metrics = model.evaluate(X_test, y_test)
    print(f"R²评分：{metrics['r2']:.4f}\n")
    # 更新最佳模型
    if metrics['r2'] > best_r2:
        best_r2 = metrics['r2']
        best_params = {
        'hidden_sizes': hidden_sizes,
        'learning_rate': lr,
        'batch_size': batch_size,
        'optimizer': optimizer.__class__.__name__
        }
        best_model = model
        # 输出最佳结果
        print("=" * 50)
        print(f"最佳超参数：{best_params}")
        print(f"最佳 R²评分：{best_r2:.4f}")
        best_model.plot