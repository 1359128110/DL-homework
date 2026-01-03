import numpy as np
from .base import Layer
from .activations import ReLU, Sigmoid
class Dense(Layer):
    """全连接层：y = W @ x + b"""
    def __init__(self, input_dim, output_dim, weight_initializer='he'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_initializer = weight_initializer
        # 权重初始化（关键：影响模型收敛速度）
        if weight_initializer == 'he':
            # He 初始化（适合 ReLU 激活）：W ~ N(0, 2/input_dim)
            self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        elif weight_initializer == 'xavier':
            # Xavier 初始化（适合 Tanh/Sigmoid）：W ~ N(0,1/(input_dim+output_dim))
            self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(1 / (input_dim + output_dim))
        else:
            raise ValueError("仅支持 he/xavier 初始化")
        # 偏置初始化（通常为 0）
        self.biases = np.zeros((1, output_dim))
        # 反向传播需用到的中间变量
        self.input = None
        self.grad_weights = None # 权重梯度
        self.grad_biases = None # 偏置梯度
    def forward(self, x):
        """前向传播：x.shape=(batch_size, input_dim) →
        output.shape=(batch_size, output_dim)"""
        self.input = x # 保存输入，用于反向传播
        return np.dot(x, self.weights) + self.biases

    def backward(self, grad_output, l2_lambda = 0.001):
        """添加 L2 正则化：梯度 += l2_lambda * weights"""
        self.grad_weights = np.dot(self.input.T, grad_output) / self.input.shape[0] + l2_lambda * self.weights
        self.grad_biases = np.mean(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input
    def update_params(self, learning_rate):
        """直接用学习率更新参数（适用于 SGD）"""
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases

class Dropout(Layer):
    """Dropout 层：防止过拟合"""
    def __init__(self, rate=0.2):
        super().__init__()
        self.rate = rate  # 丢弃概率
        self.mask = None  # 掩码（训练时生成，测试时不用）

    def forward(self, x, training=True):
        if not training:
            return x  # 测试时不丢弃，直接返回
        # 生成掩码（保留概率=1-rate）
        self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape) / (1 - self.rate)
        return x * self.mask

    def backward(self, grad_output, l2_lambda=0.0):
        return grad_output * self.mask  # 反向传播时仅保留掩码为 1 的梯度

    def update_params(self, learning_rate):
        pass

class BatchNormalization(Layer):
    """批量归一化层：加速模型收敛并提高稳定性"""
    def __init__(self, epsilon=1e-5, momentum=0.9):
        super().__init__()
        self.epsilon = epsilon  # 防止除零的小值
        self.momentum = momentum  # 移动平均的动量
        self.gamma = None  # 缩放参数
        self.beta = None  # 偏移参数
        self.running_mean = None  # 运行时均值（用于测试）
        self.running_var = None  # 运行时方差（用于测试）
        # 反向传播需要的中间变量
        self.input = None
        self.normalized_input = None
        self.mean = None
        self.var = None
        self.std = None
        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, x, training=True):
        self.input = x
        batch_size, features = x.shape

        # 初始化参数（首次前向传播时）
        if self.gamma is None:
            self.gamma = np.ones((1, features))
            self.beta = np.zeros((1, features))
            self.running_mean = np.zeros((1, features))
            self.running_var = np.ones((1, features))

        if training:
            # 计算当前批次的均值和方差
            self.mean = np.mean(x, axis=0, keepdims=True)
            self.var = np.var(x, axis=0, keepdims=True)
            self.std = np.sqrt(self.var + self.epsilon)
            
            # 归一化
            self.normalized_input = (x - self.mean) / self.std
            
            # 更新运行时均值和方差（用于测试）
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        else:
            # 测试时使用运行时均值和方差
            self.normalized_input = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # 缩放和平移
        return self.gamma * self.normalized_input + self.beta

    def backward(self, grad_output, l2_lambda=0.0):
        batch_size = grad_output.shape[0]
        
        # 计算gamma和beta的梯度
        self.grad_gamma = np.sum(grad_output * self.normalized_input, axis=0, keepdims=True)
        self.grad_beta = np.sum(grad_output, axis=0, keepdims=True)
        
        # 计算归一化输入的梯度
        grad_normalized = grad_output * self.gamma
        
        # 计算输入的梯度
        grad_var = np.sum(grad_normalized * (self.input - self.mean) * (-0.5) * np.power(self.var + self.epsilon, -1.5), axis=0, keepdims=True)
        grad_mean = np.sum(grad_normalized * (-1 / self.std), axis=0, keepdims=True) + grad_var * np.mean(-2 * (self.input - self.mean), axis=0, keepdims=True)
        grad_input = grad_normalized * (1 / self.std) + grad_var * (2 * (self.input - self.mean) / batch_size) + grad_mean / batch_size
        
        return grad_input

    def update_params(self, learning_rate):
        # 更新gamma和beta参数
        self.gamma -= learning_rate * self.grad_gamma
        self.beta -= learning_rate * self.grad_beta

class ResidualBlock(Layer):
    """残差连接块：实现高阶层间连接模式
    结构：x → Dense → BatchNormalization → ReLU → Dense → BatchNormalization → +x → ReLU
    """
    def __init__(self, input_dim, hidden_dim, output_dim, weight_initializer='he'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weight_initializer = weight_initializer
        
        # 主分支：两个全连接层 + 批量归一化
        self.dense1 = Dense(input_dim, hidden_dim, weight_initializer=weight_initializer)
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.dense2 = Dense(hidden_dim, output_dim, weight_initializer=weight_initializer)
        self.bn2 = BatchNormalization()
        
        # 跳跃连接：如果输入输出维度不同，使用1x1卷积调整维度
        if input_dim != output_dim:
            self.shortcut = Dense(input_dim, output_dim, weight_initializer=weight_initializer)
        else:
            self.shortcut = None
        
        self.relu2 = ReLU()
    
    def forward(self, x, training=True):
        # 主分支前向传播
        out = self.dense1.forward(x)
        out = self.bn1.forward(out, training=training)
        out = self.relu1.forward(out)
        out = self.dense2.forward(out)
        out = self.bn2.forward(out, training=training)
        
        # 跳跃连接
        if self.shortcut is not None:
            residual = self.shortcut.forward(x)
        else:
            residual = x
        
        # 残差连接：将输入与主分支输出相加
        out += residual
        out = self.relu2.forward(out)
        
        return out
    
    def backward(self, grad_output, l2_lambda=0.0):
        # 反向传播通过最后一个ReLU
        grad = self.relu2.backward(grad_output, l2_lambda)
        
        # 残差连接的梯度分为两部分：主分支和跳跃连接
        grad_main = grad.copy()
        grad_shortcut = grad.copy()
        
        # 主分支反向传播（注意顺序）
        grad_main = self.bn2.backward(grad_main, l2_lambda)
        grad_main = self.dense2.backward(grad_main, l2_lambda)
        grad_main = self.relu1.backward(grad_main, l2_lambda)
        grad_main = self.bn1.backward(grad_main, l2_lambda)
        grad_main = self.dense1.backward(grad_main, l2_lambda)
        
        # 跳跃连接反向传播
        if self.shortcut is not None:
            grad_shortcut = self.shortcut.backward(grad_shortcut, l2_lambda)
        
        # 总梯度 = 主分支梯度 + 跳跃连接梯度
        total_grad = grad_main + grad_shortcut
        
        return total_grad
    
    def update_params(self, learning_rate):
        # 更新主分支参数
        self.dense1.update_params(learning_rate)
        self.bn1.update_params(learning_rate)
        self.dense2.update_params(learning_rate)
        self.bn2.update_params(learning_rate)
        # 更新跳跃连接参数
        if self.shortcut is not None:
            self.shortcut.update_params(learning_rate)


class GatedResidualBlock(Layer):
    """支持跨层连接的门控残差块：
    结构：x → Dense → BatchNormalization → ReLU → Dense → BatchNormalization →
         门控机制(Sigmoid) → + 跨层连接 → ReLU
    """
    def __init__(self, input_dim, hidden_dim, output_dim, cross_connections=None, weight_initializer='he'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.cross_connections = cross_connections  # 跨层连接的输入维度列表
        self.weight_initializer = weight_initializer
        
        # 主分支：两个全连接层 + 批量归一化
        self.dense1 = Dense(input_dim, hidden_dim, weight_initializer=weight_initializer)
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.dense2 = Dense(hidden_dim, output_dim, weight_initializer=weight_initializer)
        self.bn2 = BatchNormalization()
        
        # 门控机制：使用Sigmoid控制信息流动
        self.gate_dense = Dense(input_dim, output_dim, weight_initializer=weight_initializer)
        self.sigmoid = Sigmoid()
        
        # 跳跃连接：如果输入输出维度不同，使用1x1卷积调整维度
        if input_dim != output_dim:
            self.shortcut = Dense(input_dim, output_dim, weight_initializer=weight_initializer)
        else:
            self.shortcut = None
        
        # 跨层连接：为每个跨层连接创建调整维度的层
        self.cross_shortcuts = []
        if cross_connections:
            for cross_dim in cross_connections:
                if cross_dim != output_dim:
                    self.cross_shortcuts.append(Dense(cross_dim, output_dim, weight_initializer=weight_initializer))
                else:
                    self.cross_shortcuts.append(None)
        
        self.relu2 = ReLU()
        
        # 保存跨层输入用于反向传播
        self.cross_inputs = None
    
    def forward(self, x, training=True, cross_inputs=None):
        # 保存当前输入和跨层输入
        self.input = x
        self.cross_inputs = cross_inputs
        
        # 主分支前向传播
        out = self.dense1.forward(x)
        out = self.bn1.forward(out, training=training)
        out = self.relu1.forward(out)
        out = self.dense2.forward(out)
        out = self.bn2.forward(out, training=training)
        
        # 门控机制：生成门控信号并应用
        gate = self.gate_dense.forward(x)
        gate = self.sigmoid.forward(gate)
        out = out * gate
        
        # 基础跳跃连接
        if self.shortcut is not None:
            residual = self.shortcut.forward(x)
        else:
            residual = x
        
        # 添加跨层连接
        if cross_inputs and self.cross_shortcuts:
            for i, cross_in in enumerate(cross_inputs):
                if cross_in is not None:
                    if self.cross_shortcuts[i] is not None:
                        cross_residual = self.cross_shortcuts[i].forward(cross_in)
                    else:
                        cross_residual = cross_in
                    residual += cross_residual
        
        # 残差连接：将所有残差与主分支输出相加
        out += residual
        out = self.relu2.forward(out)
        
        return out
    
    def backward(self, grad_output, l2_lambda=0.0):
        # 反向传播通过最后一个ReLU
        grad = self.relu2.backward(grad_output, l2_lambda)
        
        # 分离主分支、门控、跳跃连接和跨层连接的梯度
        grad_main = grad.copy()
        grad_gate = grad.copy()
        grad_shortcut = grad.copy()
        grad_cross_shortcuts = [grad.copy() for _ in range(len(self.cross_shortcuts))] if self.cross_shortcuts else []
        
        # 反向传播主分支
        grad_main = self.bn2.backward(grad_main, l2_lambda)
        grad_main = self.dense2.backward(grad_main, l2_lambda)
        grad_main = self.relu1.backward(grad_main, l2_lambda)
        grad_main = self.bn1.backward(grad_main, l2_lambda)
        grad_main = self.dense1.backward(grad_main, l2_lambda)
        
        # 反向传播门控机制
        gate = self.gate_dense.forward(self.input)
        gate = self.sigmoid.forward(gate)
        grad_out_gate = grad_gate * gate
        grad_gate_sig = grad_gate * self.dense2.forward(self.relu1.forward(self.bn1.forward(self.dense1.forward(self.input))))
        grad_gate_sig = self.sigmoid.backward(grad_gate_sig, l2_lambda)
        grad_gate = self.gate_dense.backward(grad_gate_sig, l2_lambda)
        
        # 反向传播基础跳跃连接
        if self.shortcut is not None:
            grad_shortcut = self.shortcut.backward(grad_shortcut, l2_lambda)
        
        # 反向传播跨层连接
        grad_cross_inputs = []
        if self.cross_shortcuts and self.cross_inputs:
            for i, (cross_shortcut, cross_input) in enumerate(zip(self.cross_shortcuts, self.cross_inputs)):
                if cross_shortcut is not None:
                    grad_cross = cross_shortcut.backward(grad_cross_shortcuts[i], l2_lambda)
                    grad_cross_inputs.append(grad_cross)
                else:
                    grad_cross_inputs.append(grad_cross_shortcuts[i])
        
        # 总梯度 = 主分支梯度 + 门控梯度 + 基础跳跃连接梯度
        total_grad = grad_main + grad_gate + grad_shortcut
        
        return total_grad, grad_cross_inputs
    
    def update_params(self, learning_rate):
        # 更新主分支参数
        self.dense1.update_params(learning_rate)
        self.bn1.update_params(learning_rate)
        self.dense2.update_params(learning_rate)
        self.bn2.update_params(learning_rate)
        # 更新门控参数
        self.gate_dense.update_params(learning_rate)
        # 更新基础跳跃连接参数
        if self.shortcut is not None:
            self.shortcut.update_params(learning_rate)
        # 更新跨层连接参数
        for cross_shortcut in self.cross_shortcuts:
            if cross_shortcut is not None:
                cross_shortcut.update_params(learning_rate)