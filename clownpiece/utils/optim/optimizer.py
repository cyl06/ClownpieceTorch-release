from clownpiece.autograd import no_grad
from clownpiece.nn import Parameter
from clownpiece.nn.init import zeros_
from clownpiece.tensor import Tensor

from typing import List, Iterable, Dict, Any, Union

class Optimizer:
    param_groups: List[Dict[str, Any]]       # list of parameter groups
    state: Dict[Parameter, Dict[str, Any]]   # mapping param_id -> optimizer state
    defaults: Dict[str, Any]                 # default hyperparams for each group

    def __init__(self, parameters: Union[Iterable[Parameter], Iterable[Dict]], defaults: Dict[str, Any]):
        """
        - `parameters`: an iterable of `Parameter` objects or a list of dicts defining parameter groups.
            - if iterable of `Parameter`, add it as the first param_group.
        - `defaults`: a dict of default hyperparameters (e.g., learning rate).
        """
        self.param_groups = []
        self.state = {}
        self.defaults = defaults
        self.id_cnt = 0
        
        parameters = list(parameters)
        
        if isinstance(parameters[0], dict):
            for param_group in parameters:
                self.add_param_group(param_group)
        else:
            self.add_param_group({'params': list(parameters)})

    def add_param_group(self, param_group: Dict[str, Any]):
        params = param_group['params']
        if isinstance(params, Parameter):
            param_group['params'] = [params]
        else:
            param_group['params'] = list(params)
        for param in param_group['params']:
            if not isinstance(param, Parameter):
                raise TypeError("optimizer can only optimize Parameters, "
                               f"but one of the params is {type(param).__name__}")
            if not hasattr(param, 'param_id'):
                param.param_id = self.id_cnt
                self.id_cnt += 1
        
        """Merge defaults into `param_group` and add to `param_groups`."""
        for k, v in self.defaults.items():
            param_group.setdefault(k, v)
        self.param_groups.append(param_group)

    def step(self):
        """Perform a single optimization step (update all parameters).
        Must be implemented by subclasses."""
        raise NotImplementedError

    def zero_grad(self, set_to_None: bool = True):
        """Reset gradients for all parameters."""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if set_to_None:
                        param.grad = None
                    else:
                        zeros_(param.grad)
  
class SGD(Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.0, damping: float = 0.0, weight_decay: float = 0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum coefficient: {momentum}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping coefficient: {damping}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        defaults = dict(lr=lr, 
                        momentum=momentum, 
                        damping=damping, 
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']                     # η
            momentum = group['momentum']         # μ
            damping = group['damping']           # γ
            weight_decay = group['weight_decay'] # λ
            
            for param in group['params']:        # θ
                if param.grad is not None:
                    grad = param.grad            # ∇L
                    
                    if weight_decay != 0.0:
                        grad += weight_decay * param
                    
                    if momentum != 0.0:
                        param_state = self.state.get(param.param_id)
                        if param_state is None:
                            buf = Tensor.zeros_like(param)
                            param_state = {'momentum_buffer': buf} 
                            self.state[param.param_id] = param_state
                        else:
                            buf = param_state['momentum_buffer']
                        
                        buf.copy_(momentum * buf + (1 - damping) * grad)
                        grad = buf
                    
                    param.copy_(param - lr * grad)

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0.0 or betas[0] >= 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if betas[1] < 0.0 or betas[1] >= 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._step_count = 0

    @no_grad()
    def step(self):
        self._step_count += 1
        t = self._step_count
        
        for group in self.param_groups:
            lr = group['lr']                     # η
            beta1, beta2 = group['betas']        # β1, β2
            eps = group['eps']                   # ϵ
            weight_decay = group['weight_decay'] # λ
            
            for param in group['params']:        # θt
                if param.grad is not None:
                    grad = param.grad            # gt
                    
                    if weight_decay != 0.0:
                        grad += weight_decay * param
                    
                    param_state = self.state.get(param.param_id)
                    if param_state is None:
                        param_state = {
                            'fir_moment': Tensor.zeros_like(param),
                            'sec_moment': Tensor.zeros_like(param),
                        }
                        self.state[param.param_id] = param_state
                    
                    m = param_state['fir_moment'] # exponentially decaying average of past gradients
                    m.copy_(beta1 * m + (1.0 - beta1) * grad)
                    
                    v = param_state['sec_moment'] # exponentially decaying average of the squared gradients
                    v.copy_(beta2 * v + (1.0 - beta2) * grad.pow(2))
                    
                    m_hat = m / (1.0 - beta1 ** t)
                    v_hat = v / (1.0 - beta2 ** t)
                    
                    param.copy_(param - lr * m_hat / (v_hat.sqrt() + eps))
