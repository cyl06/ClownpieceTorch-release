from typing import Dict, Union
from clownpiece import Tensor, zeros_like


class Parameter(Tensor):
  def __init__(self, data):
    super().__init__(data, requires_grad=True)
    

class Buffer(Tensor):
  def __init__(self, data):
    super().__init__(data, requires_grad=False)


class Module():
  _init_called: bool = False
  
  _parameters: dict[str, Parameter] = {}
  _buffers: dict[str, Tensor] = {}
  _modules: dict[str, 'Module'] = {}
  
  def __init__(self):
    self._init_called = True
    
    self._parameters = {}
    self._buffers = {}
    self._modules = {}
    
  def __setattr__(self, name: str, value):
    # remove first
    if name in self._parameters:
      del self._parameters[name]
    if name in self._buffers:
      del self._buffers[name]
    if name in self._modules:
      del self._modules[name]
    
    # add 
    if isinstance(value, Parameter):
      self._parameters[name] = value
    elif isinstance(value, Tensor):
      self._buffers[name] = value
    elif isinstance(value, Module):
      self._modules[name] = value
    else:
      super().__setattr__(name, value)
      
  def __getattribute__(self, name: str):
    if name in self._parameters:
      return self._parameters[name]
    elif name in self._buffers:
      return self._buffers[name]
    elif name in self._modules:
      return self._modules[name]
    else:
      return super().__getattribute__(name)
    
  """
    Forward
  """
    
  def forward(self, *args, **kwargs):
    raise NotImplementedError("forward method not implemented")
  
  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  """
    Parameters
  """

  def parameters(self):
    for param in self._parameters.values():
      yield param
    for module in self._modules.values():
      yield from module.parameters()

  def named_parameters(self):
    return self._named_parameters(self.__class__.__name__)
  
  def _named_parameters(self, prefix: str):
    for name, param in self._parameters.items():
      yield prefix + "." + name, param
    for name, module in self._modules.items():
      yield from module._named_parameters(prefix + "." + name)

  """
    Buffers
  """

  def buffers(self):
    for buffer in self._buffers.values():
      yield buffer
    for module in self._modules.values():
      yield from module.buffers()
  
  def named_buffers(self):
    return self._named_buffers(self.__class__.__name__)
  
  def _named_buffers(self, prefix: str):
    for name, buffer in self._buffers.items():
      yield prefix + "." + name, buffer
    for name, module in self._modules.items():
      yield from module._named_buffers(prefix + "." + name)

  """
    State Dict
  """

  def state_dict(self) -> Dict:
    return self._state_dict(self.__class__.__name__)
  
  def _state_dict(self, prefix: str) -> Dict[Union[Parameter, Buffer]]:
    state = {}
    for name, param in self._parameters.items():
      state[prefix + "." + name] = param.data
    for name, buffer in self._buffers.items():
      state[prefix + "." + name] = buffer.data
    for name, module in self._modules.items():
      state.update(module._state_dict(prefix + "." + name))
    return state
    
  def load_state_dict(self, state: Dict[Union[Parameter, Buffer]], strict: bool = True):
    self._load_state_dict(self, state=state, strict=strict)
    
  def _load_state_dict(self, state: Dict, strict: bool) -> int:
    self_name = self.__class__.__name__    
    _state = {
      k[len(self_name + "."):]: v 
      for k, v in state.items() if k.startswith(self_name + ".")
    }
    if strict and len(_state) != len(state):
      raise RuntimeError("load_state_dict: irrelevant key")
    state = _state
    
    for name, param in self._parameters.items():
      if name in state:
        param.data = state[name]
      elif strict:
        raise RuntimeError(f"load_state_dict: missing key {self_name}.{name}") 
      
    for name, buffer in self._buffers.items():
      if name in state:
        buffer.data = state[name]
      elif strict:
        raise RuntimeError(f"load_state_dict: missing key {self_name}.{name}")
      
    for name, module in self._modules.items():
      module._load_state_dict(_state)
      state = {
        k: v for k, v in state.items() if not k.startswith(name + ".")
      }
      
    if strict and len(state) != 0:
      raise RuntimeError(f"load_state_dict: irrelevant key")