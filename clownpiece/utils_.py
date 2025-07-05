# wrap x into tuple if it's not already
def wrap_tuple(x):
  return (x,) if not isinstance(x, (list, tuple)) else tuple(x)
