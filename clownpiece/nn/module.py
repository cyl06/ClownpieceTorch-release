# Core Module System

from typing import Dict, Iterable, Tuple, Union, Optional
from clownpiece import Tensor, zeros_like
from clownpiece.tensor import empty_like


class Parameter(Tensor):
  def __init__(self, data):
    super().__init__(data, requires_grad=True)
    

class Buffer(Tensor):
  def __init__(self, data):
    super().__init__(data, requires_grad=False)


class Module(object):
  training: bool
  _parameters: Dict[str, Parameter]
  _buffers: Dict[str, Buffer]
  _modules: Dict[str, "Module"]
  _init_called: bool = False
  
  def __init__(self):
    # It's a good practice to add a mechanism to enforce that:
    #   All subclasses of Module must call super().__init__ in their __init__
    #   (User often forgets to do so!)
    # For example:
    #   add a boolean variable _init_called, 
    #   and check at beginning of __setattr__ call.
    #
    # this mechanism i
    # s optional and does not account for score.
    self._init_called = True # must be initialized first XD
    self.training = True
    self._parameters = {}
    self._buffers = {}
    self._modules = {}

  def train(self, flag: bool = True):
    # set module and submodule to training = flag
    self.training = flag
    for module in self._modules.values():
      module.train(flag)

  def eval(self):
    # set module and submodule to inferencing mode
    self.train(False)

  def __setattr__(self, name, value):
    if name != "_init_called" and not getattr(self, "_init_called", False):
      raise RuntimeError(
        f"Cannot set attribute '{name}' before Module.__init__() is called. "
        f"Did you forget to call super().__init__() in '{self.__class__.__name__}'?"
      )
    
    if isinstance(value, Parameter):
      self._parameters[name] = value
    elif isinstance(value, Buffer):
      self._buffers[name] = value
    elif isinstance(value, Module):
      self._modules[name] = value
    else:
      super().__setattr__(name, value)

  def __getattr__(self, name):
    if name in self._parameters:
      return self._parameters[name]
    if name in self._buffers:
      return self._buffers[name]
    if name in self._modules:
      return self._modules[name]
    raise AttributeError(f"{type(self).__name__} has no attribute {name}")
    
  """
    Forward
  """
    
  def forward(self, *args, **kwargs):
    raise NotImplementedError("forward method not implemented")
  
  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  """
    Parameter
  """
  def register_parameter(self, name: str, param: Optional[Parameter]):
    # why does this function even exist? 
    # well, sometimes we want to register None as placeholder for disabled optioanl parameters. (e.g., bias in Linear)
    self._parameters[name] = param

  def parameters(self, recursive: bool=True) -> Iterable[Parameter]:
    # return a generator of all parameters in this module
    # yield immediate parameters first,
    # if recursive, then yield parameters from children.

    # HINT: use `yield` and `yield from` semantic
    for param in self._parameters.values():
      yield param
    if recursive:
      for module in self._modules.values():
        yield from module.parameters(True)

  def named_parameters(self, recursive: bool=True) -> Iterable[Tuple[str, Parameter]]:
    # similar to parameters, but return a name along with the parameter
    # the name is obtained by joining the recurisve attr name with '.'
    # for example
    """
      class A(Module):
        a: Parameter
        b: B

      class B(Moudle)
        c: Parameter
      
      Then, A.named_parameters() -> [
        ("a", ...),
        ("b.c", ...)
      ]
    """
    return self._named_parameters("", recursive)
  
  def _named_parameters(self, prefix: str, recursive: bool) -> Iterable[Tuple[str, Parameter]]:
    for name, param in self._parameters.items():
      yield prefix + name, param
    if recursive:
      for name, module in self._modules.items():
        yield from module._named_parameters(prefix + name + ".", recursive)

  """
    Buffer
  """

  def register_buffer(self, name: str, buffer: Optional[Buffer]):
    self._buffers[name] = buffer

  def buffers(self, recursive: bool=True) -> Iterable[Buffer]:
    for buffer in self._buffers.values():
      yield buffer
    if recursive:
      for module in self._modules.values():
        yield from module.buffers(True)

  def named_buffers(self, recursive: bool=True) -> Iterable[Tuple[str, Buffer]]:
    return self._named_buffers("", recursive)
  
  def _named_buffers(self, prefix: str, recursive: bool) -> Iterable[Tuple[str, Buffer]]:
    for name, buffer in self._buffers.items():
      yield prefix + name, buffer
    if recursive:
      for name, module in self._modules.items():
        yield from module._named_buffers(prefix + name + ".", recursive)

  """
    Modules
  """

  def register_modules(self, name: str, module: Optional["Module"]):
    self._modules[name] = module

  def modules(self, recursive: bool=True) -> Iterable["Module"]:
    for module in self._modules.values():
      yield module
    if recursive:
      for module in self._modules.values():
        yield from module.modules(True)

  def named_modules(self, recursive: bool=True) -> Iterable[Tuple[str, "Module"]]:
    return self._named_modules("", recursive)
  
  def _named_modules(self, prefix: str, recursive: bool) -> Iterable[Tuple[str, "Module"]]:
    for name, module in self._modules.items():
      yield prefix + name, module
    if recursive:
      for name, module in self._modules.items():
        yield from module._named_modules(prefix + name + ".", recursive)
    
  """
    State Dict
  """

  def state_dict(self) -> Dict:
    state = {}
    for name, param in self.named_parameters():
      state[name] = param
    for name, buffer in self.named_buffers():
      state[name] = buffer
    return state

  def load_state_dict(self, state: Dict[str, Tensor], strict: bool = True):
    redundant_weights = list(state.keys())
    missing_weights = []
    shape_mismatch = []
    
    for name, param in self.named_parameters():
      if name in state:
        if param is not None and state[name] is not None and param.shape != state[name].shape:
          shape_mismatch.append((name, param.shape, state[name].shape))
        self._set_by_path(name, state[name])
        redundant_weights.remove(name)
      elif strict:
        missing_weights.append(name)
    
    for name, buffer in self.named_buffers():
      if name in state:
        if buffer is not None and state[name] is not None and buffer.shape != state[name].shape:
          shape_mismatch.append((name, buffer.shape, state[name].shape))
        self._set_by_path(name, state[name])
        redundant_weights.remove(name)
      elif strict:
        missing_weights.append(name)
    
    if strict:
      errors = []
      if redundant_weights:
        errors.append(f"Redundant weights: {redundant_weights}")
      if missing_weights:
        errors.append(f"Missing weights: {missing_weights}")
      if shape_mismatch:
        details = ", ".join([f"{name} (expected {exp}, got {got})" for name, exp, got in shape_mismatch])
        errors.append(f"Shape mismatch for keys: {details}")
      if errors:
        raise RuntimeError("Error(s) in loading state_dict:\n" + "\n".join(errors))
  
  def _set_by_path(self, name: str, value: Tensor):
    parts = name.split(".")
    obj = self
    for part in parts[:-1]:
      obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
  
  """
    Printing
  """
  def __repr__(self) -> str:
    main_str = self.__class__.__name__
    extra = self.extra_repr()
    if extra:
      main_str += f"({extra})"
    
    child_lines = []
    for name, module in self._modules.items():
      mod_str = repr(module)
      mod_str = mod_str.replace("\n", "\n  ")
      child_lines.append(f"  ({name}): {mod_str}")

    if not child_lines:
      if not extra:
        main_str += "()"
      return main_str
    
    return main_str + "(\n" + "\n".join(child_lines) + "\n)"

  def extra_repr(self) -> str:
    return ""