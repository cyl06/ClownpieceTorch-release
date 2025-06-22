# Clownpiece-torch Week 1

Welcome to the first week of Clownpiece-torch! Our goal this week is to build a fundamental C++ tensor library. This library will serve as the cornerstone for our torch-like machine learning framework. By the end of this week, you'll have a solid understanding of how tensors are represented and manipulated in memory, which is crucial for efficient deep learning computations.

## What is a Tensor?

At its core, a **tensor** is a mathematical object that generalizes scalars, vectors, and matrices to higher dimensions.
*   A **scalar** (e.g., a single number like 5.0) can be considered a 0-dimensional tensor.
*   A **vector** (e.g., `[1.0, 2.0, 3.0]`) is a 1-dimensional tensor.
*   A **matrix** (e.g., `[[1, 2], [3, 4]]`) is a 2-dimensional tensor.

* Tensors can have any number of dimensions (also called **axes**). For instance, a 3D tensor could represent a color image (height x width x channels), and a 4D tensor could represent a batch of color images (batch_size x height x width x channels).

In deep learning, tensors are the primary data structure used to store and transform data. They can represent:
*   **Input data**: Images, text sequences, audio signals.
*   **Model parameters**: Weights and biases of neural network layers.
*   **Gradients**: Values computed during backpropagation to update model parameters.
*   **Intermediate activations**: Outputs of layers within a neural network.

Efficiently handling these multi-dimensional arrays is key to the performance of any deep learning framework.

In advanced accelerators (GPU, NPU, etc.), computation on tensors are highly optimized at hardware level to maximize parallelism.

## Decoupling Storage and Metadata

A key design principle in modern tensor libraries is the separation of **storage** from **metadata**.

*   **Storage**: This refers to the actual block of memory where the tensor's numerical data is stored. Typically, this is a contiguous, one-dimensional array (like what you would get with `new float[n]`). All elements of the tensor, regardless of its "shape," reside in this flat memory block. A **data type** (dtype) information is attached to the storage.

*   **Metadata**: This defines how we interpret the data in the storage as a multi-dimensional array. It includes:
    *   **Shape**: A list of integers defining the size of the tensor along each dimension. The total number of elements in the tensor (its **numel**) is the product of its shape dimensions
        - For example, a shape $(2, 3, 4)$ describes a 3D tensor with $2$ "planes," $3$ "rows," and $4$ "columns."; and the numel is $2 * 3 * 4 = 24$ elements.
    *   **Strides**: A list of integers indicating the number of steps to take in the 1D storage to move one unit along each dimension of the tensor. 
        - For a tensor with shape $[D_0, D_1, ..., D_{n-1}]$, the strides $[S_0, S_1, ..., S_{n-1}]$ tell us that to move from index $i$ to $i+1$ in dimension $k$, we need to advance $S_k$ elements in the underlying 1D storage.
        - If the tensor is not transposed at any dimension, then $S_k = \prod_{i=k+1}^{n-1} D_i$.
    *   **Offset**: A integer indicating the offset of the first element of the tensor in the storage. This may become non-zero if current tensor is sliced from a larger tensor.
        - To access the element with subscript $(i_0, ..., i_n)$, the storage index (the index into the 1D storage array) can be calculated as
      $$
        \text{element index} = \text{offset} + \sum_{j=0}^{n-1} i_j * S_j
      $$


Metadata combined with storage information are the essence of a tensor. That is, "storage + metadata" fully characterize a tensor.

This decoupling of storage and metadata is powerful because it allows operations like transposing, reshaping (when compatible), or taking slices of a tensor to be performed by just changing the shape, strides, and offset metadata, without actually moving or copying the data in memory. This is highly efficient.

#### Example:

Consider a 2D tensor (matrix) with shape ($2$ rows, $3$ columns) and data type float32, stored in row-major order (C style array order).

That is:
$$
A =
\begin{pmatrix}
e_0 & e_1 & e_2 \\
e_3 & e_4 & e_5
\end{pmatrix}
$$

The storage would be:

$$
\begin{cases}
  \text{data} = [e_0, e_1, e_2, e_3, e_4, e_5], \\
  \text{dtype} = \text{float}32
\end{cases}
$$

The metadata would be:

$$
\begin{cases}
\text{shape} = (2, 3) \\
\text{strides} = (3, 1) \\
\text{offset} = 0
\end{cases}
$$

Here, $\text{strides[0] = 3}$ since there are 3 element per row.

---

> Next we will introduce some tensor operations and how they motivates the decouple of storage and metadata.

## Transpose / Permutation

**Permutation** of a tensor's dimensions is a more general operation than a simple matrix transpose. If a tensor $A$ has $n$ dimensions, say $(d_0, d_1, \ldots, d_{n-1})$, a permuted tensor defined by $(p_0, p_1, \ldots, p_{N-1})$ is:

$$
  A'[(i_0, ..., i_{n-1})] = A[i_{p_0}, ..., i_{p_{n-1}}]
$$

Namely, it maps $k$-th dimension of $A'$ to $p_k$-th dimension of $A$.

In implementation, a permutation operation reorders the tensor's `shape` and `strides` metadata according to $(p_0, p_1, \ldots, p_{N-1})$. The underlying data storage is shared with the original tensor and remains unmodified.

Mathematically, it's easy to show that, with reordered strides and unchanged storage, the calculated storage offset correctly reflect the permutation operation. This technique cleverly avoids the overhead of data copying and new memory allocation.


**Example: Transposing Tensor A**

To transpose $A$ into $A^T$, we want to swap dimension 0 with dimension 1. This corresponds to the permutation $(p_0, p_1) = (1, 0)$.

The permuted tensor $A^T$ will be:
$$
  A^T=\begin{pmatrix}
  e_0 & e_3\\
  e_1 & e_4\\
  e_2 & e_5\\
  \end{pmatrix}
$$

TThe representation for $A^T$ becomes:
$$
\begin{cases}
\text{shape} = (3, 2) \\
\text{strides} = (1, 3) \\
\text{offset} = 0
\end{cases}
$$

The swap of shape dimensions according to the permutation $(1,0)$ is straightforward. The key insight is that applying the same permutation to the strides correctly reinterprets the existing data in storage to match the new dimensional order.

Let's consider accessing $A^T_{2, 0}$ (element $e_2$). Then,
$$ 
\begin{aligned}
\text{element index} &= \text{offset} + 2 \times \text{strides}[0] + 0 \times \text{strides}[1] \\
&= 0 + 2 \times 1 + 0 \times 3 \\
&= 2 
\end{aligned}
$$
This correctly locates the element $e_2$ in $[e_0, e_1, e_2, e_3, e_4, e_5]$.

## Reshaping / Viewing

Reshaping a tensor means changing its shape while preserving its total number of elements and the order of its elements. Change in number of dimension is possible during reshaping.

When we talk about the "order of elements" for reshaping, we consider the tensor's logical "flattened" version according to its current shape and strides. It's crucial to understand that this logical flattened order might not correspond to the simple linear order of data in the underlying storage if the tensor is permuted.

**Viewing vs Reshaping**

*   **Strict View (e.g., `view()`):** This operation *only* succeeds if the reshape can be achieved *without* any data copying. If a copy would be necessary because the new logical element order cannot be mapped to the existing storage via new strides, this operation raises an error.

*   **Flexible Reshape (e.g., `reshape()`):** This operation attempts to return a view without out copy. But when impossible, it makes a new contiguous tensor that reflects the reshape.


**When to copy?**

In general, a copy is required when it's impossible to reshape without reordering the elements in memory.

Yet, the exact logic to judge 'if the reshape is compatible' is hard to formulate, and few documents have details on it.

For simplicity, we allow viewing only when the tensor is **contiguous** (more specifically, row-major contiguous or C-contiguous). 

> Note that, in practice, there are incontiguous cases where viewing is still possible, if you're interested, please refer to [PyTorch's ATen source code](https://github.com/pytorch/pytorch/blob/2225231a144f182e2e1bd26a619cc1bafad49e6d/aten/src/ATen/TensorUtils.cpp#L330)

Definition of **contiguity**

- A tensor is contiguous if.f. it's elements are stored in contiguous memory, and elements' logical order correspond to physical order.

- Equivalently, a tensor is contiguous if.f.:
$$
    \forall i \in [d], \text{stride}[i] = \prod_{k=i+1}^{d} \text{shape}[k]
$$

where $d$ is the dimensionality of the tensor, and $\prod$ of empty set evaluates to $1$ by convention.

Trivially, viewing is always possible on contiguous tensors. To perform viewing on contiguous tensors, it suffice to recompute stride by $\text{stride}[i] = \prod_{k=i+1}^{d} \text{shape}[k]$, where the shape here is the proposed new shape.
 
**Example: Reshaping Tensor A**

Recall that:
$$
A =
\begin{pmatrix}
e_0 & e_1 & e_2 \\
e_3 & e_4 & e_5
\end{pmatrix}
$$
and $A$ is contiguous at this moment.

We can view $A$ into $\text{shape}=(6)$:

$$
A_1 = (e_0, e_1, e_2, e_3, e_4, e_5)
$$

with $\text{stride}=(1)$

We can view $A$ into $\text{shape}=(3, 2)$

$$
A_2 = \begin{pmatrix}
e_0 & e_1 \\
e_2 & e_3 \\
e_4 & e_5
\end{pmatrix}
$$

with $\text{stride}=(2, 1)$.

However, it is not allowed to view $A^T$ into $\text{shape}=(2, 3)$ as it's not contiguous.

---

## Broadcast

Broadcast is a helpful transformation that allows operands in tensor operations to be scaled up in some dimensions to match their shapes.

For example, you may matrix multiple a batch of matrix inputs (i.e., a 3D tensor with first dimension as batch dimension) with a weight matrix (i.e., a 2D tensor) without replicating the weight matrix along batch dimension explicitly.

Scalar + tensor is also an application of broadcast: broadcast the scalar to match the size of tensor, then perform elementwise addition.

**Formally**, a broadcast shape of two tensor is determined by:

1. Align number of dimensions by prepending $1$ s to the shape of the tensor with fewer dimensions.

2. For each dimension (starting from the last one):

- If the sizes are equal, that size is kept.

- If one of the sizes is $1$, it is broadcast to match the other size.

- If the sizes differ and neither is $1$, the tensors are not broadcastable and an error is raised.

After broadcast shape is determined, two tensors are replicated along the expanded dimensions respectively to match the shape. 

Broadcast involves no physical copy, but rather cleverly set strides of broadcasted dimensions to $0$. Thus, broadcast may lead to incontiguous tensors.


**Example:**

Let $\text{shape}_A = (3, 2, 1)$, $\text{shape}_B = (2, 1)$, then $A, B$ are broadcastable, and resulting shape is $(3, 2, 1)$

Let $\text{shape}_A = (1, 2)$, $\text{shape}_B = (3, 2)$, then $A, B$ are broadcastable, and resulting shape is $(3, 2)$


Let $\text{shape}_A = (2, 2)$, $\text{shape}_B = (3, 2)$, then $A, B$ are not broadcastable.

Let $A=(1, 2), B=\begin{pmatrix}3 & 4 \\ 5 & 6\end{pmatrix}$. For $A+B$, we broadcast $A$ to $\begin{pmatrix}1 & 2 \\ 1 & 2\end{pmatrix}$, then $A+B=\begin{pmatrix}4 & 6 \\ 6 & 8\end{pmatrix}$

---
## 📘 Additional Tutorials
* [**Intorduction To Tensor (PyTorch doc)**](https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html):  An beginner guide that thoroughly explores the usage of torch.Tensor class. Have a look if you are unfamiliar with tensor in PyTorch.

* [**PyTorch internals (Blog by ezyang)**](https://blog.ezyang.com/2019/05/pytorch-internals): A detailed deep‑dive and conceptual tour of PyTorch’s core tensor abstraction—including metadata like layout, device, dtype, and stride—and how these underpin view creation, slicing, and the broader C++ kernel and autograd layers .

---

# Code Guide

---

## Prerequisites

Before getting started, ensure your environment meets the following requirements:

---

### 1. **Linux Environment**

A Linux-based environment is required for this project.

* **WSL (Windows Subsystem for Linux)** is acceptable.
* **Important:** Do **not** place the project directory inside the Windows file system (e.g., under `/mnt/c/...`).
  File access across the Windows-Linux boundary in WSL is slow because it goes through a virtualized network layer.
  Instead, place the project in your WSL’s native Linux file system (e.g., somewhere under `~/`).

---

### 2. **C++ Toolchain: `g++` (with C++17) and `CMake`**

Ensure the following tools are installed:

* `g++` with support for **C++17**
* `cmake` (version 3.10 or later is recommended)

You can install them using your Linux distribution’s package manager:

**For Debian/Ubuntu:**

```bash
sudo apt update
sudo apt install build-essential cmake
```

**For Fedora/RHEL:**

```bash
sudo dnf install gcc-c++ cmake
```

---

### 3. **Conda (for Python environment management)**

[Install Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) if you haven’t already.

Then, create and activate a virtual environment for the project:

```bash
conda create -n ct python=3.10
conda activate ct
```

Choose python interpreter at the bottom-right corner of vscode window, and next time it will activate the desired environment automatically at terminal start-up.

---

### 4. **Python Dependencies**

With the `ct` environment activated, install the required Python packages:

```bash
# Install pybind11 (for C++/Python binding)
pip install pybind11
```

To avoid include errors like `#include <pybind11/pybind11.h> not found`, add the pybind11 include path under conda environment to the VSCode C++ extension:

1. Find the pybind11 include path in your conda env
> python -m pybind11 --includes

It returns something like:
> -I/path/to/conda/env/include -I/path/to/conda/env/lib/pythonX.Y/site-packages/pybind11/include
2. Add `/path/to/conda/env/lib/pythonX.Y/site-packages/pybind11/include` to [include path setting](vscode://settings/C_Cpp.default.includePath) here.

---

### 5. **PyTorch for Reference**

PyTorch is not required for the build or runtime of this project, but is part of the grader library. **You must install PyTorch to run the grader**.

Besides, you may refer to PyTorch's behavior for a better understanding of what you need to implement.

Refer to the [official installation guide](https://pytorch.org/get-started/locally/) to install PyTorch tailored to your CUDA or CPU configuration:

```bash
pip install torch torchvision torchaudio
```

**If you encounter any problems with pybind, don't hesitate to consult MasterFHC**

---

## Code Structure Overview

```bash
clownpiece
|--autograd # This is for week2, you don't need it now
|--tensor # The C++ tensor library you're going to implement
| |- meta.h
| |- meta.cc
| |- tensor.h
| |- tensor.cc
| |- tensor.py
| |- tensor_pybind.cc
|- __init__.py
|- setup.py
|- tensor.py # Bindings for week2 
|... # rest of clownpiece
tests
|--week1 # testcases for week1
| |- grade_part{1-8}.py
| |- grade_all.py
| |- grade_comprehensive.ipynb
| |- graderlib.py
```
*   **`meta.h` / `meta.cc`**: These files handle fundamental definitions and utilities.
    *   `meta.h` defines common types used throughout the tensor library, such as `dtype` (data type, e.g., float), `shape_t` (for tensor dimensions), `stride_t` (for tensor strides). It also declares random number generation functions.
    *   `meta.cc` provides the implementations for the random number generation functions declared in `meta.h`.
    You can make changes to these files if necessary, for example, to add new type aliases or utility functions.

*   **`tensor.h`**: This is the header file for the `Tensor` class itself. It will contain the declaration of the `Tensor` class, including its member variables (like storage pointer, shape, strides, offset) and method signatures for operations like transpose, reshape, arithmetic operations, etc. You can make changes to this file, for instance, to add new methods or friend function declarations.

*   **`tensor.cc`**: This is the core implementation file where you will write the logic for the `Tensor` class methods declared in `tensor.h`.

*   **`tensor.py`**: This file defines a Python `Tensor` class. This Python class will act as a wrapper around the C++ `Tensor` object that you build. In general, you should not need to make significant changes to this file, as it primarily delegates calls to the C++ backend.

*   **`tensor_pybind.cc`**: This file uses the `pybind11` library to expose your C++ `Tensor` class and its methods to Python. It creates the necessary bindings so that the `tensor.py` wrapper can call your C++ code. You generally won't need to modify this file.

*   **`setup.py`**: This script is used to compile your C++ source files (`tensor.cc`, `meta.cc`, `binding.cc`) into a shared library (a Python extension module, e.g., a `.so` file on Linux or `.pyd` on Windows). This allows Python to import and use your C++ tensor library.

> Make sure you have read at least meta.h/meta.c/tensor.h before you start.

---

---

## How to compile and test your code?

run the following command to recompile your code
> cd clownpiece && pip install -e .

then, go to directory `tests/week1` to run test scripts or directly run 
> python ../tests/week1/grade_part{i}.py

There is one test script `grade_part{i}.py` for each part of code below, As a lifesaver, you can run `grade_all.py` to test all `grade_part{i}.py` at once!

There is another comprehensive test script: `grade_comprehensive.ipynb`.

You must pass both `grade_all.py` and `grade_comprehensive.ipynb`.

To see full traceback in debug mode, use `DEBUG=1 python grade_part{i}.py`.

**Important Remark: For all invalid cases and exceptions (out_of_range, invalid_argument, etc.), your code must throw instance of `std::runtime_error` to be properly graded with test scripts**

>> Hint: The coding order presented below is a suggestion, not a strict requirement. You may find it beneficial to skip ahead to implement utility functions from later sections if they can assist with earlier parts. Read the tensor.h thoroughly to grasp the outline of our tensor library.

---

Next, let's get down to coding. 

Our interfaces are supposed to be compatible with pytorch unless otherwise specified. Please refer to well-documented PyTorch if you are uncertain of functions' behavior. (link is provided at entry for corresponding functions.)

## Part I: constructors, assignments, destructors, and item()

```cpp
  // zero dim tensor with no data
  Tensor();
  
  // zero dim tensor with a scalar data
  Tensor(dtype value);
  
  // tensor with given shape, with data uninitialized
  explicit Tensor(const shape_t& shape);

  // tensor with given shape, with data initialized to given value
  explicit Tensor(const shape_t& shape, dtype value);
  
  // tensor with given shape, with data initialized by a generator function
  explicit Tensor(const shape_t& shape, std::function<dtype()> generator);
  
  // tensor with given shape, with a vector as underlying data
  explicit Tensor(const shape_t& shape, const vec<dtype>& data);
  
  // tensor constructed from metadata + storage
  explicit Tensor(const shape_t& shape, const stride_t& stride, int offset, 
  Storage storage);
  
  // shallow copy a tensor
  Tensor(const Tensor& other);

  // shallow copy a tensor
  Tensor& operator=(const Tensor& other);

  // only valid for singleton tensor
  Tensor& operator=(dtype value);

  // destructor
  ~Tensor();


  /*
    convert to dtype value
  */
  // only valid for singleton tensor
  dtype item() const;
```

All functions here are straightforward without further explanation.

## Part II: utils, clone, make contiguous, copy_, scatter_

```cpp
  int numel() const;
  int dim() const;
  veci size() const;
  int size(int dim) const;
  bool is_contiguous() const;

  Tensor clone() const;
  Tensor contiguous() const;
  Tensor copy_(const Tensor& other) const;
  Tensor scatter_(int dim, const Tensor& index, const Tensor& src) const;
```

---
> **int numel() const**:

Return number of elements in the tensor. [link](https://docs.pytorch.org/docs/stable/generated/torch.numel.html#torch.numel)
>> Hint: you must handle cases where dimension of tensor is zero. The zero dimension tensor can be either scalar or completely empty.

---
> **int dim() const**:

Return number of dimension of the tensor. [link](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.dim.html)

---
> **veci size() const** or **int size(int dim) const**:

If `dim` is not provided, return the entire shape vector.

Otherwise, return an int holding size of given dimension. The `dim` follows python style index, and may be negative. Throw `std::runtime_error` if out of range. [link](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.size.html#torch.Tensor.size)

---
> **bool is_contiguous() const**:

Return if the tensor is contiguous in memory. [link](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html#torch.Tensor.is_contiguous)

>> Hint: precompute `is_contiguous_`, `numel_`, `shape_prod_` (i.e., strides in contiguous case, for faster indexing conversion) at constructor or assignment to reduce overhead.

---
> **Tensor clone() const**:
  
Return a new contiguous tensor with underyling data copied. [link](https://docs.pytorch.org/docs/stable/generated/torch.clone.html#torch.clone)

In newer version of PyTorch, clone does not necessarily return a contiguous tensor, but in our implement, please make it contigous.

---
> **Tensor contiguous() const**:

Returns a contiguous tensor containing the same data as self tensor. If the self tensor is contiguous, a view of itself is returned. Otherwise, a cloned tensor is returned. [link](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html#torch.Tensor.contiguous)

---
> **Tensor copy_(const Tensor& other) const;**: 

Copy other's data into the underlying storage and return itself. The corresponding method in pytorch is `Tensor.copy_`. [link]()

>> Hint: writing utilities like `int offset_at(int n)`, `dtype& data_at(int n)` is effective. They calcuate the offset in storage to the $i$-th element in flat logic order.

---
> **Tensor scatter_(int dim, const Tensor& index, const Tensor& src) const**

Scatter `src` tensor along given dimention `dim` at `index`.

For example:
```python
result[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
result[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
result[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
```

`index` and `src` must match the shape of self tensor after removal of `dim`.

## Part III: subscriptor

```cpp
  Tensor operator[](const vec<slice_t>& slices) const;
  Tensor operator[](slice_t slice) const;

  Tensor operator[](const veci& indices) const
  Tensor operator[](int index) const;
```

---
> **Tensor Tensor::operator[](const vec<slice_t>& slices) const**
> **Tensor operator[](slice_t slice) const**

Return a sub-tensor view of self tensor, with given argument slice.

The slice follows python style, and may contains negative indices. 
- If the slice is empty (say, begin $\ge$ end), those dimenstions should be set to empty slice (e.g. [0: 0]), but do not raise exceptions. 
- If the slice is out of range, you should clip the slice into range, but not raise exceptions. For example, slice = (2, 4), while size = 3, then slice should be clipped to (2, 3) instead.
- You need not to support 'step' feature (which is the third argument to slice in python), and can assume step=1.


If the slices are shorter than tensor's dimension, then perform sli cing from leading dimenstions.

Dimenstions should **not** be sequeeze after slicing.

Example
```python
>> A = Tensor([[0, 1], 
               [2, 3]])

>> A[0:1]
Tensor([[0, 1]])
>> A[0:-1]
Tensor([[0, 1]])
>> A[0:-100]
Tensor([[]])
```

---
> **Tensor operator[](const veci& indices) const**
> **Tensor operator[](int index) const**

Accesses elements or a sub-tensor using specific integer indices.

The indices follow python style, and may contains negative indices. If out of range, throw `std::runtime_error`.

If the indices are shorter than tensor's dimension, then perform indexing from leading dimenstions.

Dimensions should be squeezed after indexing.

Example:
```python
>> A = Tensor([[0, 1], 
               [2, 3]])
>> A[1]
Tensor([2, 3])
(not Tensor([[2, 3]]))
```


> Remark: when binding to python, we only use the singleton version of `int index` or `slice_t slice`. Real python indexing can be a list mixed with both index and slice. Handling these type variance in C++ is tedious, so we avoid that.
> However, implement vector version of `veci index` and `vec<slice_t> slices` may help you in shape manipulation functions.

---

## Part IV: Element-wise Binary and Unary Operators

```cpp
  // Unary negation
  Tensor operator-() const;

  // Element-wise addition
  friend Tensor operator+(const Tensor& lhs, const Tensor& rhs);

  // Element-wise subtraction
  friend Tensor operator-(const Tensor& lhs, const Tensor& rhs);

  // Element-wise multiplication
  friend Tensor operator*(const Tensor& lhs, const Tensor& rhs);

  // Element-wise division
  friend Tensor operator/(const Tensor& lhs, const Tensor& rhs);

  // Comparison
  friend Tensor operator==(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator!=(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator<(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator<=(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator>=(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator>(const Tensor& lhs, const Tensor& rhs);

  // Other uniary mathematical operations
  Tensor sign() const;

  Tensor abs() const;

  Tensor sin() const;

  Tensor cos() const;

  Tensor tanh() const;

  Tensor clamp(dtype min, dtype max) const;

  Tensor log() const;

  Tensor exp() const;
  
  Tensor pow(dtype exponent) const;

  Tensor sqrt() const;
```

### ! uniary mathematical operations have non-member variants functions.

---
> **Tensor operator-() const** and uniary mathematical operators:

Returns a new tensor with the element-wise negation/math op of the self tensor.
1.  The returned tensor will have the same shape as self tensor.
2.  This operation creates a new tensor with new storage.

---
> **friend Tensor operator+(const Tensor& lhs, const Tensor& rhs)**

> **friend Tensor operator-(const Tensor& lhs, const Tensor& rhs)**

> **friend Tensor operator\*(const Tensor& lhs, const Tensor& rhs)**

> **friend Tensor operator/(const Tensor& lhs, const Tensor& rhs)**:

> **friend Tensor operator==(const Tensor& lhs, const Tensor& rhs);**

> **friend Tensor operator!=(const Tensor& lhs, const Tensor& rhs);**

> **friend Tensor operator<(const Tensor& lhs, const Tensor& rhs);**

> **friend Tensor operator<=(const Tensor& lhs, const Tensor& rhs);**

> **friend Tensor operator>=(const Tensor& lhs, const Tensor& rhs);**

> **friend Tensor operator>(const Tensor& lhs, const Tensor& rhs);**

Perform element-wise binary operations (addition, subtraction, multiplication, division).
1.  **Broadcasting**: These operations support broadcasting. If `lhs` and `rhs` have different but compatible shapes, they are broadcast to a common shape before the operation. If shapes are not broadcastable, throw `std::runtime_error`.
    *   A scalar operand (e.g., `tensor + 5`) is handled by implicitly converting the scalar to a 0-dimensional tensor (by constructor), which then participates in broadcasting.
2. **Apply Operator**: The operation is applied element-wise to the broadcasted tensors to create a new tensor. (for comparison operator, 1 if the comparison holds, otherwise 0.)

---

---

> Tensor sign() const;
$$
  \operatorname{sign}(x) = \begin{cases}
  1 & x>0 \\
  0 & x=0 \\
  -1 & x < 0 \\
  \end{cases}
$$
> Tensor abs() const;

> Tensor sin() const;

> Tensor cos() const;

> Tensor tanh() const;

> Tensor clamp(dtype min, dtype max) const;

  $$\operatorname{clamp(a, lower, upper) = \min\{upper, \max\{lower, a\}\}}$$

> Tensor log() const;

> Tensor exp() const;

$$
  \exp(x) = e^x
$$

> Tensor pow(dtype exponent) const;

$$
  \operatorname{pow}(x, \text{exponent}) = x^{\text{exponent}}
$$

> Tensor sqrt() const;

>> Hint: implement general functions to have unified interface and reduce coding workload.

>> `Tensor apply_unary_op(std::function<dtype(dtype)> op)`

>> `Tensor apply_binary_op(std::function<dtype(dtype, dtype)> op)` 

---

## Part V: Matrix Multiplication

```cpp
  friend Tensor matmul(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator^(const Tensor& lhs, const Tensor& rhs); // Equivalent to matmul
```

The rule for matmul is a bit complicated, please refer to [pytorch's document](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul):

- if both operands are 1-D, a dot product 0-D scalar tensor is returned
- if lhs is 1D, while rhs is at least 2-D, then a 1 is padded to the lhs' dimensions to make it 2-D. The padded dimension is removed at return.
- if rhs is 1D, while lhs has at least 2-D, then a 1 is padded to the rhs' dimensions to make it 2-D, and then rhs is transposed. The padded dimension is removed at return.
- if both operands are at least 2-D, then last 2 dimensions are treated as matrix dimension, and remaining leading dimensions are broadcasted.
  - for example, $L=(i\times 1 \times n \times m), R = (j \times m \times l)$, then return will be $(i \times j \times n \times l)$.

>> Hint: Viewing intermediate tensor to shape $(\{-1, N\})$ often makes logic easier and clear. Also, reordering intermediate axis to last dimension with transpose may help exploiting faster contiguous access.

---

## Part VI: Reduction and Normalization Operations

```cpp
  Tensor sum(int dim, bool keepdims=false) const;

  std::pair<Tensor, Tensor> max(int dim, bool keepdims=false) const; // Returns (values, indices)
  
  Tensor softmax(int dim) const;
```

---
> **Tensor sum(int dim, bool keepdims=false) const**

Returns the sum along dimension `dim`. [link (2nd variant)](https://pytorch.org/docs/stable/generated/torch.sum.html)
1.  `dim`: The dimension to reduce. Negative indexing is supported.
2.  `keepdims`: If `true`, the output tensor is of the same dimensionality as `self`, with dimension `dim` having size 1. If `false` (default), the dimension `dim` is squeezed.

Example:
```python
>> A = Tensor([[0, 0], [1, 1]])
>> A.sum(0, keepdims=True)
Tensor([[0], [2]])
>> A.sum(1, keepdims=False)
Tensor([1, 1])
```

> Remark: you only need to implement single dimension version `sum`, and we bind multi-dimension version in python for you. Besides, the default `dim` argument is `None` in python that sums over all dimension and produces a singleton tensor.

---
> **std::pair<Tensor, Tensor> max(int dim, bool keepdims=false) const; // Returns (values, indices)**

Returns a pair of tensors: the maximum values and the indices of these maximum values along the dimension `dim`. [link argmax](https://docs.pytorch.org/docs/stable/generated/torch.argmax.html#torch.argmax), [link max](https://docs.pytorch.org/docs/stable/generated/torch.max.html#torch.max)

The indices are primarily for autograd purpose.

---
> **Tensor softmax(int dim) const**:

Returns the softmax along dimension `dim`. [link](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch.nn.functional.softmax) 

$$
  \operatorname{softmax}({a_i}, \{a_j\}_{j\in[n]}) = \dfrac{\exp(a_i)}{\sum_{j=1}^n \exp(a_j)}
$$

As for tensor:

$$
  \operatorname{softmax}(A) = \dfrac{A.\text{exp}()}{A.\text{exp}().\text{sum}(\text{dim}, \text{keepdims}=true)}
$$

---

## Part VII: Shape Manipulation

```cpp
  Tensor permute(veci p) const;
  Tensor transpose(int dim1, int dim2) const;

  Tensor reshape(const shape_t& purposed_shape, bool copy=false) const;
  Tensor view(const shape_t& purposed_shape) const;

  Tensor narrow(int dim, int start, int length, bool copy=false) const;
  vec<Tensor> chunk(int chunks, int dim) const;

  vec<Tensor> split(int dim, int split_size) const;
  vec<Tensor> split(int dim, veci split_sections) const;

  static Tensor stack(const vec<Tensor>& inputs, int dim=0);
  static Tensor cat(const vec<Tensor>& inputs, int dim=0);

  Tensor squeeze(int dim) const;
  Tensor unsqueeze(int dim) const;

  Tensor broadcast_to(const shape_t& shape) const;
  static std::pair<Tensor, Tensor> broadcast(const Tensor& lhs, const Tensor& rhs);
  static vec<Tensor> broadcast(const vec<Tensor>& tensors);

```

---
> **Tensor permute(veci p) const**:

Returns a view of the original tensor with its dimensions permuted according to `p`. [link](https://pytorch.org/docs/stable/generated/torch.permute.html)

- `p`: A `veci` (vector of ints) specifying the new order of dimensions. `p` may contain only part of the dimensions, if so, the other dimensions remain unchanged. Negative indexing is supported

---
> **Tensor transpose(int dim1, int dim2) const**:

Returns a view of the original tensor with dimensions `dim1` and `dim2` swapped. [link](https://pytorch.org/docs/stable/generated/torch.transpose.html)
- `dim1`, `dim2`: The dimensions to be swapped. Negative indexing is supported. Two dims may be equal.


---
> **Tensor reshape(const shape_t& purposed_shape, bool copy=false) const**:

Returns a tensor with the specified `purposed_shape` which have the same data as self tensor.

- whether to copy
  1. if viewing is possible, and `copy=false`, return viewing
  2. otherwise, return a contiguous new tensor.

- shape deduction:
  1. at most one '-1' is supported in `purposed_shape`, which will be duduced from numel and other specified dimensions.

---
> **Tensor view(const shape_t& purposed_shape) const**:

Returns a tensor with the specified `purposed_shape` which have the same data as self tensor without copying.

For simplicity, we only allow viewing if the tensor is contiguous. If not, throw `std::runtime_error`.

---
> **Tensor narrow(int dim, int start, int length, bool copy=false) const**:

Return a sliced tensor at `dim`, with `[start:start+length]`. Return a view if `copy=false`; otherwise, return a contiguous new tensor. [link](https://docs.pytorch.org/docs/stable/generated/torch.narrow.html#torch.narrow)

---
> **vec<Tensor> chunk(int chunks, int dim) const**

[link](https://docs.pytorch.org/docs/stable/generated/torch.chunk.html#torch.chunk)

Attempts to split a tensor into the specified number of `chunks`. Each chunk is a view of the input tensor.

If the tensor size along the given dimension `dim` is divisible by `chunks`, all returned chunks will be the same size. If the tensor size along the given dimension dim is not divisible by `chunks`, all returned chunks will be the same size, except the last one. If such division is not possible, this function may return fewer than the specified number of chunks.

---
> **vec<Tensor> split(int dim, int split_size) const** & **vec<Tensor> split(int dim, veci split_sections) const**

[link](https://docs.pytorch.org/docs/stable/generated/torch.split.html#torch.split)

Overload 1:
  - tensor will be split into equally sized chunks (if possible). Last chunk will be smaller if the tensor size along the given dimension `dim` is not divisible by `split_size`.

Overload 2:
  -  tensor will be split into `split_sections.size()` chunks with sizes in `dim` according to `split_size_or_sections`.

> **static Tensor stack(const vec<Tensor>& inputs, int dim=0)**

  Stack the `inputs` along a new dimension, inserted before `dim`. [link](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack)
  
  - `dim`: Ranges from 0 to dim() (inclusive). Negative indexing supported (relative to the new number of dimensions). 
  
  - `inputs`: All inputs must be of the same shape

> **static Tensor cat(const vec<Tensor>& inputs, int dim=0)**

  Concatenate the `inputs` along an existing dimension `dim`. [link](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat)

  - `inputs`: All inputs must be of the same shape except the concatentated dimension.


> **Tensor squeeze(int dim) const**:

Returns a tensor with the dimension `dim` of size 1 removed. [link](https://pytorch.org/docs/stable/generated/torch.squeeze.html)


1.  `dim`: The dimension to squeeze. If the shape at `dim` is not $1$, raise error. Negative indexing supported.

2.  View: Returns a new tensor (view) sharing the same storage.

> **Tensor unsqueeze(int dim) const**:

Returns a new tensor with a dimension of size 1 inserted before `dim`. [link](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html)


1.  `dim`: The index at which to insert the singleton dimension (before the dim). Ranges from `0` to `dim()` (inclusive). Negative indexing supported (relative to the new number of dimensions).
2.  View: Returns a new tensor (view) sharing the same storage.

---

> **Tensor broadcast_to(const shape_t& shape) const**:

  broadcast the tensor to proposed `shape`. The shape must be broadcastable. [link](https://docs.pytorch.org/docs/stable/generated/torch.broadcast_to.html#torch.broadcast_to)

> **static std::pair<Tensor, Tensor> broadcast(const Tensor& lhs, const Tensor& rhs)** &

> **static vec<Tensor> broadcast(const vec<Tensor>& tensors)**;

  broadcast a pair or list of tensors to common shape. The broadcast mechanism is detailed before. [link](https://docs.pytorch.org/docs/stable/generated/torch.broadcast_tensors.html#torch.broadcast_tensors)

---

## Part VIII: Other Helper Constuctors:

```cpp
Tensor to_singleton_tensor(dtype value);

Tensor ones(const shape_t& shape);
Tensor ones_like(const Tensor& ref);

Tensor zeros(const shape_t& shape);
Tensor zeros_like(const Tensor& ref);

Tensor randn(const shape_t& shape);
Tensor randn_like(const Tensor& ref);

Tensor empty(const shape_t& shape);
Tensor empty_like(const Tensor& ref);

Tensor arange(dtype start, dtype end, dtype step);

Tensor range(dtype start, dtype end, dtype step);

Tensor linspace(dtype start, dtype end, int num_steps);
```


### ! ones_like, zeros_like, randn_like, empty_like have member variants functions.

---

> **Tensor to_singleton_tensor(dtype value, int dim)**

Creates a tensor of shape `[1] * dim` containing a single `value`.

---

> **Tensor ones(const shape_t& shape)**

Returns a tensor filled with the scalar value `1`.

---

> **Tensor ones_like(const Tensor& ref)**

Returns a tensor filled with ones, having the same shape as `ref`.

---

> **Tensor zeros(const shape_t& shape)**

Returns a tensor filled with the scalar value `0` of the specified shape

---

> **Tensor zeros_like(const Tensor& ref)**

Returns a tensor of filled with zeros, having the same shape as `ref`.

---

> **Tensor randn(const shape_t& shape)**

Returns a tensor of the specified shape with elements drawn from a standard normal distribution (`mean=0`, `std=1`).

---

> **Tensor randn_like(const Tensor& ref)**

Returns a tensor with random values drawn from the standard normal distribution, having the same shape as `ref`.

---

> **Tensor empty(const shape_t& shape)**

Returns an uninitialized tensor of the specified shape.

---

> **Tensor empty_like(const Tensor& ref)**

Returns an uninitialized tensor, having the same shape as `ref`

---

> **Tensor arange(dtype start, dtype end, dtype step)**

Returns a 1-D tensor of size  
\[
\left\lfloor \frac{\text{end} - \text{start}}{\text{step}} \right\rfloor
\]  
with values from the interval `[start, end)` taken with common difference `step`, starting from `start`. [link](https://docs.pytorch.org/docs/stable/generated/torch.arange.html#torch.arange)

- **start**: Start value (inclusive).
- **end**: End value (exclusive).
- **step**: Increment value between elements.
- **Returns**: A 1D tensor from `start` (inclusive) to `end` (exclusive) with step `step`.

---

> **Tensor range(dtype start, dtype end, dtype step)**

Returns a 1-D tensor of size  
\[
\left\lfloor \frac{\text{end} - \text{start}}{\text{step}} \right\rfloor + 1
\]  
with values from `start` to `end`, inclusive, using step `step`. [link](https://docs.pytorch.org/docs/stable/generated/torch.range.html#torch.range)

- **start**: Start value (inclusive).
- **end**: End value (inclusive).
- **step**: Increment value between elements.
- **Returns**: A 1D tensor from `start` to `end` (inclusive) with step `step`.

---

> **Tensor linspace(dtype start, dtype end, int num_steps)**

Returns a 1-D tensor of `num_steps` values linearly spaced between `start` and `end`, inclusive. [link](https://docs.pytorch.org/docs/stable/generated/torch.linspace.html#torch.linspace)

- **start**: Start of the interval.
- **end**: End of the interval.
- **num_steps**: Number of points to generate.
- **Returns**: A 1D tensor of evenly spaced values from `start` to `end`.

---
---

# Optional Challenge: Parallelize Tensor Computation

### Why Parallelism?

Modern CPUs and GPUs are equipped with multiple cores that allow computations to be executed concurrently. However, naive implementations of tensor operations in Python often run sequentially, leaving most computational resources underutilized. By parallelizing operations, we can reduce runtime and improve scalability—especially for compute-intensive tasks like matrix multiplication and large tensor transformations.

This challenge explores different parallelization strategies to speed up tensor operations.

---

### Parallelization Strategies

You are encouraged to implement and evaluate **three different approaches** to parallelize tensor operations:

#### 1. Thread-per-Operation

This approach is straightforward to implement but may incur high overhead due to excessive thread creation, especially for small workloads.

* **Pros**: Easy to understand and implement.
* **Cons**: Poor scalability; thread spawning overhead becomes a bottleneck.


For basic tutorial of C++ standard library multithreading, please refer to [this link](https://www.geeksforgeeks.org/cpp/multithreading-in-cpp/).

#### 2. Implicit Parallelism

Use [OpenMP](https://www.openmp.org/) or other implicit parllelism library to auto-parallelize loops over tensor data. OpenMP handles thread spawning, work partitioning, and synchronization transparently.

* **Pros**: Efficient, widely supported, minimal boilerplate.
* **Cons**: Limited flexibility.


#### 3. Thread Pool

Create a fixed-size thread pool and dispatch work in chunks. This reduces thread overhead and can reuse threads across many operations.

* **Pros**: Balanced performance and flexibility. Lower thread spawning overhead.
* **Cons**: Requires more design effort for synchronization and task scheduling.

You may utilize existing lightweighted C++ threadpool like [BS::thread_pool](https://github.com/bshoshany/thread-pool).

---

### Evaluation Task

Apply each method to the following workloads at different input size, and report performance:

1. **Element-wise Operation**: e.g., `y = x * 2 + 3`
2. **Matrix Multiplication**: `C = A @ B` where $A \in \mathbb{R}^{N \times M}, B \in \mathbb{R}^{M \times K}$

### Report Requirements
* Clearly describe:

  * How you implemented and optimized each method
  * Your test setup (CPU specs, dataset size, time measuring method, thread number etc.)
* Provide benchmark results:

  * Plot runtime vs. input size for all 3 methods and 2 wrokloads
  * Compare against a baseline sequential implementation

* Analyze:

  * Which method performed best in general and why?
  * At what point does parallelization overhead outweigh the benefits? Based on this knowledge, can you vary number of thread according to task scale to achieve better performance?
  * For which operation(s) is parallelization most beneficial?

If you find a certain method too difficult to implement, you may include a description of your partial attempt in the report. Partial credit for the optional challenge will be awarded based on the completeness and effort of your implementation.

We acknowledge that the workload for the first week is heavier than average. You are welcome to revisit and complete the optional challenges in future weeks when time permits.

> ! Note: please use compile time flag to control which method is enabled, and at submission, please ensure default compile behavior is sequential version, so that TAs can run your program without any OpenMP or custom thread pool.

---

## Submit Your Homework

First, make sure that you passed the `grade_all.py` as well as `grade_comprehensive.ipynb`.

Then, you should write a detailed **report** under `docs/week1`, to describe the challenges you have encountered, how did you solve them, and what are the takeaways. (Also, attach the grader summary part from output of `grade_all.py`). This report accounts for part of your score.

Finally zip the entire project folder into lab-week1.zip, and submit to canvas.

Make sure that the TAs can run the `grade_all.py` and `grade_comprehensive.ipynb` (or just leave the running results of ipynb as it is) and find your report from your submission.