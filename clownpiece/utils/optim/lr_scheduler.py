from .optimizer import Optimizer
from typing import List, Callable

class LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        # You need to store base_lrs as to allow closed-form lr computation; you may assume that, optimizer does not add new param groups after initializing the LRSchduler.
        self.optimzer = optimizer
        self.last_epoch = last_epoch
        
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
        self._last_lr = self.base_lr.copy()
        
        self.step()

    def get_lr(self) -> List[float]:        
        # Calculate new learning rates for each param_group.
        # To be implemented by subclasses with a closed-form formula based on self.last_epoch.
        raise NotImplementedError

    def step(self, epoch: int = None):
        # Advance to `epoch` (or next epoch if None), then update optimizer.param_groups:
        # 1. Update self.last_epoch 
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        # 2. Compute new LRs via self.get_lr()
        new_lrs = self.get_lr()
        # 3. Assign each param_group['lr'] = corresponding new LR
        for group, lr in zip(self.optimzer.param_groups, new_lrs):
            group['lr'] = lr
        self._last_lr = new_lrs.copy()

    def get_last_lr(self) -> List[float]:
        # Return the most recent learning rate for each param_group.
        return self._last_lr
      
    
class LambdaLR(LRScheduler):
    """
    Lambda learning rate scheduler.
    Applies a user-defined function to the learning rate.
    """
    def __init__(self, optimizer, lr_lambda: Callable[[int], float], last_epoch: int = -1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [base_lr * self.lr_lambda(self.last_epoch) for base_lr in self.base_lr]
    
class ExponentialLR(LRScheduler):
    """
    Exponential learning rate scheduler.
    Multiplies the learning rate by a factor every epoch.
    """
    
    def __init__(self, optimizer, gamma: float = 0.1, last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [base_lr * (self.gamma ** self.last_epoch) for base_lr in self.base_lr]
        
class StepLR(LRScheduler):
    """
    Step learning rate scheduler.
    Decreases the learning rate by a factor every `step_size` epochs.
    """
    
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [base_lr * (self.gamma ** (self.last_epoch // self.step_size)) for base_lr in self.base_lr]
    
