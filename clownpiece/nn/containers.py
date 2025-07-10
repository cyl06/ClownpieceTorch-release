# Sequential, ModuleList, ModuleDict

from typing import Iterable, Dict, Tuple
from clownpiece.nn.module import Module
class Sequential(Module):
  
  def __init__(self, *modules: Module):
    super().__init__()
    for i, module in enumerate(modules):
      self.register_modules(str(i), module)
    self._sequence = list(modules)

  def forward(self, input):
    for module in self._sequence:
      input = module(input)
    return input


class ModuleList(Module):
  
  def __init__(self, modules: Iterable[Module] = None):
    # hint: try to avoid using [] (which is mutable) as default argument. it may lead to unexpected behavor.
    # also be careful to passing dictionary or list around in function, which may be modified inside the function.
    super().__init__()
    self._list = []
    if modules is not None:
      for module in modules:
        self.append(module)

  def __add__(self, other: Iterable[Module]):
    return ModuleList(list(self._list), list(other))

  def __setitem__(self, index: int, value: Module):
    self._list[index] = value
    self.register_modules(str(index), value)

  def __getitem__(self, index: int) -> Module:
    return self._list[index]

  def __delitem__(self, index: int):
    del self._list[index]
    del self._modules[str(index)]

  def __len__(self):
    return len(self._list)

  def __iter__(self) -> Iterable[Module]:
    return iter(self._list)

  def append(self, module: Module):
    index = len(self._list)
    self._list.append(module)
    self.register_modules(str(index), module)

  def extend(self, other: Iterable[Module] = None):
    if other is not None:
      for module in other:
        self.append(module)

class ModuleDict(Module):
  
  def __init__(self, dict_: Dict[str, Module] = None):
    super().__init__()
    if dict_ is not None:
      for name, module in dict_.items():
        self.register_modules(name, module)

  def __setitem__(self, name: str, value: Module):
    self.register_modules(name, value)

  def __getitem__(self, name: str) -> Module:
    return self._modules[name]

  def __delitem__(self, name: str):
    del self._modules[name]

  def __len__(self):
    return len(self._modules)

  def __iter__(self) -> Iterable[str]:
    return iter(self._modules)
  
  def keys(self) -> Iterable[str]:
    return self._modules.keys()

  def values(self) -> Iterable[Module]:
    return self._modules.values()

  def items(self) -> Iterable[Tuple[str, Module]]:
    return self._modules.items()

  def update(self, dict_: Dict[str, Module] = None):
    if dict_ is not None:
      for name, module in dict_.items():
        self[name] = module