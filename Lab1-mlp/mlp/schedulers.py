import numpy as np


class LearningRateScheduler:
    """学习率调度器基类"""
    
    def __init__(self, initial_lr):
        self.initial_lr = initial_lr
    
    def get_lr(self, epoch):
        """根据当前 epoch 获取学习率"""
        raise NotImplementedError


class CosineAnnealingLR(LearningRateScheduler):
    """余弦退火学习率调度器：
    lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(epoch / T_max * pi))
    """
    
    def __init__(self, initial_lr, T_max, eta_min=0):
        super().__init__(initial_lr)
        self.T_max = T_max  # 周期
        self.eta_min = eta_min  # 最小学习率
    
    def get_lr(self, epoch):
        # 余弦退火公式
        return self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (1 + np.cos(epoch / self.T_max * np.pi))


class StepLR(LearningRateScheduler):
    """阶梯式学习率调度器：每 step_size 个 epoch 学习率乘以 gamma"""
    
    def __init__(self, initial_lr, step_size, gamma=0.1):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self, epoch):
        return self.initial_lr * (self.gamma ** (epoch // self.step_size))


class ExponentialLR(LearningRateScheduler):
    """指数衰减学习率调度器：lr = initial_lr * gamma^epoch"""
    
    def __init__(self, initial_lr, gamma):
        super().__init__(initial_lr)
        self.gamma = gamma
    
    def get_lr(self, epoch):
        return self.initial_lr * (self.gamma ** epoch)