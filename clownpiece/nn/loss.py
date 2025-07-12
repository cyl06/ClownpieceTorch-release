# MSE, CrossEntropy

from clownpiece.nn.module import Module
from clownpiece import Tensor


# loss.py
class MSELoss(Module):
  def __init__(self, reduction: str = 'mean'):
    super().__init__()
    self.reduction = reduction
    if reduction not in ['mean', 'sum']:
      raise ValueError(f"Unsupportede reduction: {reduction}. Only 'mean' and 'sum' are supported.")
    
  def forward(self, input: Tensor, target: Tensor) -> Tensor:
    diff = input - target
    squared_diff = diff * diff
    
    if self.reduction == 'mean':
      return squared_diff.reshape(-1).mean(-1)
    else:
      return squared_diff.reshape(-1).sum(-1)

class CrossEntropyLoss(Module):
  def __init__(self, reduction: str = 'mean'):
    super().__init__()
    self.reduction = reduction
    if reduction not in ['mean', 'sum']:
      raise ValueError(f"Unsupportede reduction: {reduction}. Only 'mean' and 'sum' are supported.")
  
  def forward(self, logits: Tensor, target: Tensor) -> Tensor:
    # logits is of shape (..., num_class)
    # target is of shape (...), and it's value indicate the index of correct label

    # You need to ensure your implement is differentiable under our autograd engine.
    # However, you can assume target has requires_grad=False and ignore the gradient flow towards it.
    
    log_sum_exp = logits.exp().sum(dim=-1, keepdims=True).log()
    log_probs = logits - log_sum_exp
    
    batch_size = logits.shape[0]
    losses = []
    for i in range(batch_size):
      target_i = int(target[i].item())
      losses.append(-log_probs[i, target_i].unsqueeze())
    loss = Tensor.cat(losses)

    if self.reduction == 'mean':
      return loss.mean(-1)
    else:
      return loss.sum(-1)