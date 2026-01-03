import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os

from mlp.layers import Dropout


class MLP:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        # 训练历史（用于可视化）
        self.history = {
            'train_loss': [],
            'val_loss': []
        }

    def add_layer(self, layer):
        """添加层（按顺序添加，输入层→隐藏层→输出层）"""
        self.layers.append(layer)

    def set_loss(self, loss):
        """设置损失函数"""
        self.loss = loss

    def set_optimizer(self, optimizer):
        """设置优化器"""
        self.optimizer = optimizer

    # def forward(self, x):
    #     """前向传播：计算模型输出"""
    #     output = x
    #     for layer in self.layers:
    #         output = layer.forward(output)
    #     return output
    def forward(self, x, training=True):
        output = x
        # 保存各层的输出，用于跨层连接
        layer_outputs = [x]  # 添加输入层的输出到layer_outputs中
        
        for layer in self.layers:
            if isinstance(layer, Dropout):
                output = layer.forward(output, training=training)
            elif hasattr(layer, 'cross_connections') and layer.cross_connections:
                # 对于支持跨层连接的门控残差块，提供跨层输入
                cross_inputs = []
                for cross_dim in layer.cross_connections:
                    # 查找具有匹配维度的层输出
                    found = False
                    for prev_output in reversed(layer_outputs):
                        if prev_output.shape[1] == cross_dim:
                            cross_inputs.append(prev_output)
                            found = True
                            break
                    if not found:
                        cross_inputs.append(None)
                # 不需要反转cross_inputs，因为我们已经按照cross_connections的顺序收集了
                output = layer.forward(output, training=training, cross_inputs=cross_inputs)
            else:
                output = layer.forward(output)
            
            # 保存当前层的输出，用于后续层的跨层连接
            layer_outputs.append(output)
        return output

    def backward(self, y_pred, y_true, l2_lambda=0.0):
        """反向传播：计算梯度（添加L2正则化参数）"""
        if self.loss is None:
            raise ValueError("请先调用 set_loss 设置损失函数")
        grad = self.loss.backward(y_pred, y_true)
        
        # 反向遍历层（从输出层到输入层）
        for layer in reversed(self.layers):
            if hasattr(layer, 'cross_connections') and layer.cross_connections:
                # 处理门控残差块的反向传播，它返回两个值：总梯度和跨层输入的梯度
                grad, _ = layer.backward(grad, l2_lambda)
            else:
                # 普通层的反向传播
                grad = layer.backward(grad, l2_lambda)
        
        return grad

    def update(self):
        """更新模型参数"""
        if self.optimizer is None:
            raise ValueError("请先调用 set_optimizer 设置优化器")
        for layer in self.layers:
            self.optimizer.update(layer)

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_data=None, early_stopping=False, patience=10, scheduler=None):
        """ 训练模型
        :param X_train: 训练特征 (n_samples, n_features)
        :param y_train: 训练标签 (n_samples,) 或 (n_samples, 1)
        :param epochs: 迭代次数
        :param batch_size: 批次大小
        :param validation_data: 验证集 (X_val, y_val)
        :param early_stopping: 是否启用早停
        :param patience: 早停容忍度（多少个epoch验证损失未改善则停止）
        :param scheduler: 学习率调度器（可选）
        :return: 训练历史
        """
        n_samples = X_train.shape[0]
        y_train = y_train.reshape(-1, 1)  # 统一形状为(n_samples, 1)
        
        # 早停相关变量
        best_val_loss = float('inf')
        best_weights = None
        no_improvement_count = 0

        for epoch in range(1, epochs + 1):
            # 更新学习率（如果有调度器）
            if scheduler is not None:
                current_lr = scheduler.get_lr(epoch - 1)  # epoch从0开始计算
                # 根据优化器类型设置学习率
                if hasattr(self.optimizer, 'lr'):
                    self.optimizer.lr = current_lr
                elif hasattr(self.optimizer, 'learning_rate'):
                    self.optimizer.learning_rate = current_lr
            
            # 打乱训练数据（避免顺序依赖）
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            train_loss = 0.0
            n_batches = n_samples // batch_size

            # 批次训练
            for i in range(n_batches):
                # 取批次数据
                start = i * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # 前向传播
                y_pred = self.forward(X_batch)

                # 计算损失
                batch_loss = self.loss.forward(y_pred, y_batch)
                train_loss += batch_loss

                # 反向传播（添加L2正则化）
                self.backward(y_pred, y_batch, l2_lambda=0.001)

                # 更新参数
                self.update()

            # 计算平均训练损失
            avg_train_loss = train_loss / n_batches
            self.history['train_loss'].append(avg_train_loss)

            # 计算验证损失（如有验证集）
            val_loss = None
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val = y_val.reshape(-1, 1)
                y_val_pred = self.forward(X_val)
                val_loss = self.loss.forward(y_val_pred, y_val)
                self.history['val_loss'].append(val_loss)

                # 早停检查
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = self._save_weights()
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        
                    if no_improvement_count >= patience:
                        print(f"\n早停：验证损失在 {patience} 个epoch内未改善")
                        print(f"最佳验证损失：{best_val_loss:.4f}")
                        self._load_weights(best_weights)
                        break

            # 每 10 个 epoch 打印一次日志
            if epoch % 10 == 0:
                lr_info = f" | LR: {current_lr:.6f}" if scheduler is not None else ""
                if val_loss is not None:
                    print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}{lr_info}")
                else:
                    print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train_loss:.4f}{lr_info}")

        return self.history
    
    def _save_weights(self):
        """保存当前模型权重（用于早停）"""
        weights = []
        
        def save_layer_weights(layer):
            """递归保存层权重"""
            layer_weights = []
            # 如果是残差块，保存其内部所有层的权重
            if hasattr(layer, 'dense1') and hasattr(layer, 'dense2'):
                layer_weights.append(save_layer_weights(layer.dense1))
                layer_weights.append(save_layer_weights(layer.dense2))
                if layer.shortcut is not None:
                    layer_weights.append(save_layer_weights(layer.shortcut))
                return layer_weights
            # 保存直接包含weights和biases的层
            elif hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                return (np.copy(layer.weights), np.copy(layer.biases))
            # 其他层返回None
            else:
                return None
        
        for layer in self.layers:
            weights.append(save_layer_weights(layer))
        return weights
    
    def _load_weights(self, saved_weights):
        """加载保存的模型权重（用于早停）"""
        
        def load_layer_weights(layer, saved_layer_weights):
            """递归加载层权重"""
            if saved_layer_weights is None:
                return
            # 如果是残差块，递归加载其内部所有层的权重
            if hasattr(layer, 'dense1') and hasattr(layer, 'dense2'):
                load_layer_weights(layer.dense1, saved_layer_weights[0])
                load_layer_weights(layer.dense2, saved_layer_weights[1])
                if layer.shortcut is not None and len(saved_layer_weights) > 2:
                    load_layer_weights(layer.shortcut, saved_layer_weights[2])
            # 加载直接包含weights和biases的层
            elif hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                layer.weights, layer.biases = saved_layer_weights
        
        for i, layer in enumerate(self.layers):
            if saved_weights[i] is not None:
                load_layer_weights(layer, saved_weights[i])

    def evaluate(self, X_test, y_test):
        """ 评估模型性能
        :return: (test_loss, metrics) → metrics 包含 MAE、R²
        """
        X_test = X_test.reshape(-1, X_test.shape[-1])
        y_test = y_test.reshape(-1, 1)
        y_pred = self.forward(X_test).reshape(-1, 1)

        # 计算损失和指标
        test_loss = self.loss.forward(y_pred, y_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {
            'mae': mae,
            'r2': r2
        }
        return test_loss, metrics

    def predict(self, X):
        """预测新数据"""
        return self.forward(X).reshape(-1)  # 输出形状为 (n_samples,)

    def plot_history(self, save_path='results/training_curve.png'):
        """绘制训练曲线（训练损失 vs 验证损失）"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(self.history['train_loss'], label='Train Loss', linewidth=2)

        if 'val_loss' in self.history and self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')

        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('Training & Validation Loss Curve', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_predictions(self, X_test, y_test, save_path='results/predictions.png'):
        """绘制真实值 vs 预测值散点图"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        y_pred = self.predict(X_test)

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, s=50)

        # 绘制理想预测线（y=x）
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Prediction (y=x)')

        plt.xlabel('True Housing Price', fontsize=12)
        plt.ylabel('Predicted Housing Price', fontsize=12)
        plt.title('True vs Predicted Housing Prices', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()