#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#include "tensor.h"
#include <cmath>
#include <string>



namespace at {

  /*
    utils for printing
  */
  // print vector int
  std::ostream& operator<<(std::ostream& os, const shape_t& shape) {
    os << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
      os << shape[i];
      if (i < shape.size() - 1)
        os << ", ";
    }
    os << ")";
    return os;
  }

  int print_tensor_data_recursive(std::ostream& os, const Tensor& tensor, int dim_index, int data_index, std::string prefix) {
    if (tensor.dim() == 0) {
      if (tensor.numel() == 0)
        os << "[]";
      else
        os << tensor.data_at(0);
      return 0;
    }
    os << "[";
    if (dim_index == tensor.dim() - 1 || tensor.dim() == 0) {
      for (int i = 0; i < tensor.size(dim_index); ++i) {
        os << tensor.data_at(data_index++);
        if (i < tensor.size(dim_index) - 1)
            os << ", ";
      }
    } else {

      for (int i = 0; i < tensor.size(dim_index); ++i) {
        if (i > 0)
          os << "\n" << prefix;
        data_index = print_tensor_data_recursive(os, tensor, dim_index + 1, data_index, prefix + " ");
        if (i < tensor.size(dim_index) - 1)
          os << ",";
      }
    }
    os << "]";
    return data_index;
  }

  std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(\n  shape=" << tensor.shape_ << ", strides=" << tensor.stride_ << "\n  data={\n";
    std::string prefix = "    ";
    os << prefix;
    print_tensor_data_recursive(os, tensor, 0, 0, prefix + " ");
    os << "\n  }\n)\n";
    return os;
  }

  /*
    Begin your implement here !
  */

  dtype& Tensor::data_at(int index) const {}

  /*
    constructors and assignments
  */
  Tensor::Tensor() {}
  Tensor::Tensor(dtype value) {}
  Tensor::Tensor(const shape_t& shape) {}
  Tensor::Tensor(const shape_t& shape, dtype value) {}
  Tensor::Tensor(const shape_t& shape, std::function<dtype()> generator) {}
  Tensor::Tensor(const shape_t& shape, const vec<dtype>& data) {}
  Tensor::Tensor(const shape_t& shape, const stride_t& stride, int offset, Storage storage) {}

  Tensor::Tensor(const Tensor& other) {}

  Tensor& Tensor::operator=(const Tensor& other) {}

  Tensor& Tensor::operator=(dtype value) {}

  /* 
    destructor
  */
  Tensor::~Tensor() {}


  /*
    convert to dtype value
    only valid for singleton tensor
  */
  dtype Tensor::item() const {}

  /*
    utils
  */

  int Tensor::numel() const {}

  int Tensor::dim() const {}

  veci Tensor::size() const {}

  int Tensor::size(int dim) const {}

  bool Tensor::is_contiguous() const {}


  /*
    clone, make contiguous, copy_ and scatter
  */
  Tensor Tensor::clone() const {}

  Tensor Tensor::contiguous() const {}

  Tensor Tensor::copy_(const Tensor& other) const {}

  Tensor Tensor::scatter_(int dim, const Tensor& index, const Tensor& src) const {}


  /*
    subscriptor
  */

  Tensor Tensor::operator[](const vec<slice_t>& slices) const {}

  Tensor Tensor::operator[](slice_t slice) const {}

  Tensor Tensor::operator[](const veci& index) const {}

  Tensor Tensor::operator[](int index) const {}

  /*
    operators
  */
  Tensor Tensor::operator-() const {}

  Tensor operator+(const Tensor& lhs, const Tensor& rhs) {}
  
  Tensor operator-(const Tensor& lhs, const Tensor& rhs) {}

  Tensor operator*(const Tensor& lhs, const Tensor& rhs) {}

  Tensor operator/(const Tensor& lhs, const Tensor& rhs) {}
  
  Tensor operator==(const Tensor& lhs, const Tensor& rhs) {}

  Tensor operator!=(const Tensor& lhs, const Tensor& rhs) {}
  Tensor operator<(const Tensor& lhs, const Tensor& rhs) {}

  Tensor operator<=(const Tensor& lhs, const Tensor& rhs) {}

  Tensor operator>=(const Tensor& lhs, const Tensor& rhs) {}

  Tensor operator>(const Tensor& lhs, const Tensor& rhs) {}

  /*
    matrix multiplication
  */
  Tensor matmul(const Tensor& lhs, const Tensor& rhs) {}

  Tensor operator^(const Tensor& lhs, const Tensor& rhs) {
  }

  /*
    other mathematical operations
  */
  Tensor Tensor::sign() const {}

  Tensor Tensor::abs() const {}
  Tensor abs(const Tensor& tensor) {}

  Tensor Tensor::sin() const {}
  Tensor sin(const Tensor& tensor) {}

  Tensor Tensor::cos() const {}
  Tensor cos(const Tensor& tensor) {}
  Tensor Tensor::tanh() const {}
  Tensor tanh(const Tensor& tensor) {}

  Tensor Tensor::clamp(dtype min, dtype max) const {}

  Tensor clamp(const Tensor& tensor, dtype min, dtype max) {}

  Tensor Tensor::log() const {}

  Tensor log(const Tensor& tensor) {}

  Tensor Tensor::exp() const {}

  Tensor exp(const Tensor& tensor) {}

  Tensor Tensor::pow(dtype exponent) const {}

  Tensor pow(const Tensor& tensor, dtype exponent) {}

  Tensor Tensor::sqrt() const {}

  Tensor sqrt(const Tensor& tensor) {}

  Tensor Tensor::sum(int dim, bool keepdims) const {}

  Tensor sum(const Tensor& tensor, int dim, bool keepdims) {}

  std::pair<Tensor, Tensor> Tensor::max(int dim, bool keepdims) const {}

  std::pair<Tensor, Tensor> max(const Tensor& tensor, int dim, bool keepdims) {}

  Tensor Tensor::softmax(int dim) const {}
  Tensor softmax(const Tensor& tensor, int dim) {}

  /*
    helper constructor
  */

  Tensor Tensor::ones_like() const {}
  Tensor Tensor::zeros_like() const {}
  Tensor Tensor::randn_like() const {}
  Tensor Tensor::empty_like() const {}

  /*
    shape manipulation
  */

  Tensor Tensor::permute(veci p) const {}

  Tensor Tensor::transpose(int dim1, int dim2) const {}

  Tensor Tensor::reshape(const shape_t& purposed_shape, bool copy) const {}

  Tensor Tensor::view(const shape_t& purposed_shape) const {}

  Tensor Tensor::narrow(int dim, int start, int length, bool copy) const {}

  vec<Tensor> Tensor::chunk(int chunks, int dim) const {}

  vec<Tensor> Tensor::split(int dim, int split_size) const {}
  vec<Tensor> Tensor::split(int dim, veci split_sections) const {}

  Tensor Tensor::stack(const vec<Tensor>& inputs, int dim) {}

  Tensor Tensor::cat(const vec<Tensor>& inputs, int dim) {}

  Tensor Tensor::squeeze(int dim) const {}

  Tensor Tensor::unsqueeze(int dim) const {}

  Tensor Tensor::broadcast_to(const shape_t& shape) const {}

  std::pair<Tensor, Tensor> Tensor::broadcast(const Tensor& lhs, const Tensor& rhs) {}
  vec<Tensor> Tensor::broadcast(const vec<Tensor>& tensors) {}



  /*
    helper constructors
  */
  Tensor to_singleton_tensor(dtype value, int dim) {}

  Tensor ones(const shape_t& shape) {}
  Tensor ones_like(const Tensor& ref) {}

  Tensor zeros(const shape_t& shape) {}
  Tensor zeros_like(const Tensor& ref) {}

  Tensor randn(const shape_t& shape) {}
  Tensor randn_like(const Tensor& ref) {}

  Tensor empty(const shape_t& shape) {}
  Tensor empty_like(const Tensor& ref) {}

  Tensor arange(dtype start, dtype end, dtype step) {}

  Tensor range(dtype start, dtype end, dtype step) {}

  Tensor linspace(dtype start, dtype end, int num_steps) {}
  
  /*
    Week3 adds-on
  */
  Tensor Tensor::mean(int dim, bool keepdims) const {}

  Tensor Tensor::var(int dim, bool keepdims, bool unbiased) const {}
};
