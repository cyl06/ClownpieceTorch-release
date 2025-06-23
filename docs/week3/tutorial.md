# Clownpiece-torch Week 3

In Week 2, we built a powerful autograd engine capable of tracking computations and automatically calculating gradients. While this is the core of modern deep learning frameworks, writing complex models using only raw tensor operations can be cumbersome and disorganized. This week, we will build a **Module** system, inspired by PyTorch's `torch.nn.Module`, to bring structure, reusability, and convenience to our model-building process.

The module system provides a way to encapsulate parts of a neural network into reusable components. It handles the management of learnable parameters, sub-modules, and stateful buffers, allowing you to define complex architectures in a clean, object-oriented way.

A simple example in PyTorch illustrates the concept:
```python
import torch
import torch.nn as nn

# Define a custom network by subclassing nn.Module
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define layers as attributes. They are automatically registered.
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    # Define the forward pass
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Instantiate the model
model = SimpleNet(input_size=784, hidden_size=128, output_size=10)
print(model)

# The module system makes it easy to inspect all parameters
print("\nNamed Parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```
Outputs:
```python
SimpleNet(
  (layer1): Linear(in_features=784, out_features=128, bias=True)
  (activation): ReLU()
  (layer2): Linear(in_features=128, out_features=10, bias=True)
)

Named Parameters:
layer1.weight: torch.Size([128, 784])
layer1.bias: torch.Size([128])
layer2.weight: torch.Size([10, 128])
layer2.bias: torch.Size([10])
```

As you can see, the `nn.Module` base class provides a clean structure and automatically tracks all the learnable parameters within the nested layers.

## Unifying Computation and State

The first design philosophy is the **unified managment of tightly coupled parts**. 

A neural network layer isn't just a single function; it's a stateful computation. It has a defined transformation (the computation) and it has internal variables that persist across calls (the state). Therefore, module comes to help by organizing of *computation* and *states* together.

In implement, module elegantly organizes this into three fundamental components:

-   **Forward Pass**: It defines the transformation the module applies to its inputs. You can think of it as the mathematical function the layer represents, like a linear transformation or a convolution. The forward takes both user specified inputs and module's internal states to produce the outputs.

-   **Parameters**: These represent the learnable state of the module, often referred to as **weights**. When you 'train' a model, you are optimizing these parameters to achieve some objective (i.e., minizing a loss function, or maximizing downstream task's accuracy). Parameters account for the majority of state in a typical deep learning model.

-   **Buffers**: These represent the non-learnable state. Sometimes a module needs to keep track of data that isn't a learnable parameter, such as the running mean and variance in a batch normalization layer. They are saved along with the parameters, but they are not updated by the optimizer during backpropagation. You will only see buffers in few special modules.

Clearly, forward pass defines the computation, while parameters and buffers form the state.

> Both parameters and buffers can change across calls, so the term *non-learnable* does **NOT** imply *constant* or *immutable*. It's more of a model structrual concept: whether they can be optimized, or only for temporary storage purpose.

---

#### Example

Let's consider a `Linear` module, which performs $y=x@W^T+b$ (we will explain the reason for transpose later).

Its forward pass might be like:
```python
class Linear(Module):
  W: Tensor # shape [ouput_channel * input_channel]
  b: Tensor # shape [ouput_channel]

  def forward(self, x: Tensor) -> Tensor: # shape [... * input_channel] ->  [... * output_channel]
    W, b = self.W, self.b
    y = x @ W.transpose() + b 
    return y
```
where `@`, `transpose`, and `+` are traced by autograd engine, and dispatched into tensor library at runtime.

`self.W, self.b` are parameters, and there are no buffers in `Linear`.

---

## Modular and Hierarchical Organization

### Modularity $\to$ Simpilicity, Reusability and Flexibility

The second core design philosophy emphasizes **modularity**. Just as individual layers encapsulate their own logic and variables, these self-contained modules can be nested and connected to form intricate network architectures.

By breaking down a complex neural network into smaller, manageable modules, the design process becomes much simpler. Instead of dealing with a monolithic block of code, you can focus on developing and testing individual components. 

Moreover, modules are highly reusable. 
- Mainstream DL frameworks offer highly-optimized, well-tested implementations for common modules like Linear, Conv2d, BatchNorm. 
- Even domain-specific modules can be reused. FlashAttention, RotaryEmbedding are widely adopted in different transformer models.

Modularity also introduces immense flexibility. Suppose you're an enthusiastic researcher with an idea to alter the structure of an existing state-of-the-art model. With a modular design, you can easily swap out or introduce new components, without having to rebuild the entire network from scratch or understanding other parts' implementation detail. This iterative approach is crucial for innovation and experimentation in deep learning, especially when models are getting more and more complicated nowadays.

---
### Hierarchy $\to$ Ease of Design and System Managment

Beyond modularity, the module system is inherently **hierarchical**, which is excellent news for system designers. Higher-level modules are composed of smaller, more basic modules, but never the other way around. 

However, from a functional standpoint, there's no noticeable difference between a basic layer and a complex block; they both remain unified under the *module abstraction* with great modularity.

> Modularity and hierarchy usually contradict each other, so this example is interesting.

With hierarchical structure, we can conceptualize a module's composition as a tree, where clear parent-child relationships are defined. 

This allows a parent module to manage the states of all its children. This centralized state management is beneficial for saving, updating, or restoring the entire module's state.

---

#### Example: Unfolding a GPT-like Model

Consider a **GPT-model**. Its module structure will be like:

```python
GPTModel
├── Embedding # Projects input IDs into hidden space
├── Positional Encoding # Adds positional information
└── Transformer Blocks # Manipulates the hidden states
    ├── Transformer Block 1
    │   ├── Multi-Head Attention # Attention mechanism
    │   │   ├── Linear # for Q, K, V projections
    │   │   └── Linear # for output projection
    │   ├── Layer Normalization # Layer Norm
    │   └── Feed-Forward Network # FFN
    │       ├── Linear
    │       └── Activation
    └── Transformer Block 2
        ├── Multi-Head Attention
        │   ├── Linear
        │   └── Linear
        ├── Layer Normalization
        └── Feed-Forward Network
            ├── Linear
            └── Activation
    └── ... (repeated Transformer Blocks)
```

There might also be a `LM Head` layer at the end to project hidden states back into the probability space of output IDs, depending on the downstream task.

**Modularity**

All these modules are implemented separately elsewhere and then assembled to form the `GPTModel`. Beyond `GPTModel` itself, these `Transformer Blocks` can be reused in other architectures like ViT, Llama, etc., perhaps with slight modifications to adapt to specific contexts.

**Hierarchy**

The `GPTModel` is the top-level module containing `Embedding`, `Positional Encoding`, and the collection of `Transformer Blocks`. 

- Each `Transformer Block` encapsulating `Multi-Head Attention`, `Layer Normalization`, and a `Feed-Forward Network`. 
  - Further, `Multi-Head Attention` and `Feed-Forward Network` are themselves composed of simpler `Linear` and `Activation` modules. 

> Note that the tree hierarchy does not imply sequential recursive execution of childeren modules. The exact computation logic is defined by user in forward function and may form a complex DAG.

> Yet, it's true that we register child module in the order they are executed by convention, and, most of the time, they are sequential.

---

## Layered System Design:

When working on the code this week, you may find that the autograd engine and tensor library hide the most complexitiy of underyling computation and backward tracing. Modules feel like a simple wrapper around the autograd system with state management.

Yes,that's exactly why we design it this way: seperating the system functionalities into distinct layers, where higher layer only relies on lower layers, and mostly, only its adjacent layer. This brings great similicity for both design and implement.

Module system is completely agnoistic to how autograd engine or tensor library work under the hood -- it just assumes they will do what they promise to do properly. Conversely, autograd engine or tensor library need not care how module system operates. 

> Though, from a designer's perspective, it is important to design good interfaces, with which higher layer can utilize lower layer efficiently and easily. This always requires a global view of the system.

Meanwhile this layered abstration is perfect in our problem, it is usually over-simplified or ideal in more complex systems. In those cases, engineers takes a middle ground between monolithic and layered design (i.e., modularity when possible). You will learn that in next year's operating system course.




## 📘 Additional Tutorials

Understanding how `nn.Module` works is fundamental to using (and then implementing) any modern deep learning library effectively. 

* [**`torch.nn.Module` Official Documentation**](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
  The definitive reference for all `nn.Module` functionality.

* [**Building Models with PyTorch**](https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)
  A beginner-friendly tutorial on how to define and use `nn.Module` to build networks.

* [**Saving and Loading Models in PyTorch**](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
  A comprehensive guide on using `state_dict` to persist and restore model state.

---

# Code Guide

---

This week's focus leans more towards user-friendly system design rather than intricate low-level engineering. The elegance of a well-structured system lies in its simplicity for the end-user. So, to grasp what "user-friendly" means:

### **We highly recommend you getting familiar with PyTorch's module system before proceeding to code your own!**

Please refer to Addtional Tutorials above.

---

## Code Structure Overview

```bash
clownpiece
|-nn
| |- activations.py
| |- containers.py
| |- init.py
| |- layers.py
| |- loss.py
| |- module.py
|-...
```

- The `module.py` holds the core definition of abstract class `Module`, centralizing common functionalities for all modules.

- The `init.py` holds utility to initialize parameters in different probabilistic ways. 

- Other files contain concrete modules of a specific type suggested by the file name.

We'll follow these steps for implementation:

1. First, implement the core `Module` class in `module.py`, including
  - parameter and buffer management
  - state_dict save and load
  - \_\_repr\_\_ method to visualize module structure
  - pre/post forward hooks
2. Next, create several simplest concrete modules to rigorously test if the `Module`'s fundamental functionalities are correctly working.
3. Then, develop the `init.py` utilities for parameter initialization.
4. Finally, complete the implementation of other specific modules in `activations.py`, `layers.py`, `containers.py`, and `loss.py`.

5. Try out what you module system with two traditional DL taks.

Due to the workload restriction and incompleteness of our autograd engine and tensor library, we can only explore a small portion of common modules.

> Anyway, a complete DL framework cannot be built from scratch in only weeks; don't be disappointed; We will build some interesting application with what we have!🤗

---