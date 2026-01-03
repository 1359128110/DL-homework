class Layer:
    """层基类：所有层需实现 forward/backward 方法"""
    def forward(self, x):
        raise NotImplementedError("子类必须实现 forward 方法")
    def backward(self, grad_output):
        raise NotImplementedError("子类必须实现 backward 方法")
    def update_params(self, learning_rate):
        """无参数层（如激活函数）无需实现"""
        pass


class Activation(Layer):
    """激活函数基类：保存输入用于反向传播"""
    def __init__(self):
        self.input = None