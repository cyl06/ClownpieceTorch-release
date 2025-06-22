from contextlib import contextmanager

# Global variable to track gradient computation state
_grad_enabled = True

def is_grad_enabled():
    """Returns whether gradient tracking are currently enabled."""
    return _grad_enabled

@contextmanager
def no_grad():
    """
    Context-manager that disables gradient calculation.
    
    Within this context, gradients will not be calculated, and `requires_grad` flags
    will be ignored. This can be used to improve performance when you don't need
    gradients, such as during inference.
    
    Example:
        ```python
        with no_grad():
            # Computations here don't track gradients
            result = model(input_data)
        ```
    """
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = previous

@contextmanager
class set_grad_enabled:
    """
    Context-manager that sets gradient calculation to on or off.
    
    Parameters:
        mode (bool): Flag whether to enable gradients (True) or disable (False)
    
    Example:
        ```python
        with set_grad_enabled(False):
            # Computations here don't track gradients
            result = model(input_data)
        ```
    """
    def __init__(self, mode):
        self.mode = mode
        self.prev = is_grad_enabled()
        
    def __enter__(self):
        global _grad_enabled
        _grad_enabled = self.mode
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _grad_enabled
        _grad_enabled = self.prev