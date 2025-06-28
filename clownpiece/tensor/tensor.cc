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

  dtype& Tensor::data_at(int index) const {
    if (index < 0 || index >= numel_) {
      throw std::runtime_error("out of range (Func data_at)");
    }
    
    if (is_contiguous_) {
      return storage_[offset_ + index];
    }
    
    int element_index = offset_;
    for (int i = 0; i < dim_; i++) {
      element_index += index / shape_prod_[i] * stride_[i];
      index %= shape_prod_[i];
    }
    
    return storage_[element_index];
  }
  
  // auxiliary initialization
  veci Tensor::calc_prod() {
    numel_ = 1;
    dim_ = shape_.size();
    veci prod(dim_);
    for (int i = dim_ - 1; i >= 0; i--) {
      prod[i] = numel_;
      numel_ *= shape_[i];
    }
    return prod;
  }
  
  int Tensor::init_set() {
    is_contiguous_ = true;
    stride_ = shape_prod_;
    return 1;
  }
  
  int Tensor::init_copy() {
    if (stride_.size() != dim_) {
      throw std::runtime_error("invalid construct (Tensor copy)");
    }
    is_contiguous_ = true;
    for (int i = dim_ - 1; i >= 0; i--) {
      if (stride_[i] != shape_prod_[i]) {
        is_contiguous_ = false;
      }
    }
    return 2;
  }

  /*
    constructors and assignments
  */
  // zero dim tensor with no data
  Tensor::Tensor() : shape_(shape_t()), shape_prod_(calc_prod()), stride_(stride_t()), offset_(0), storage_(Storage()) { init_set(); }
  
  // zero dim tensor with a scalar data
  Tensor::Tensor(dtype value) : shape_(shape_t()), shape_prod_(calc_prod()), stride_(stride_t()), offset_(0), storage_(Storage(numel_, value)) { init_set(); }
  
  // tensor with given shape, with data uninitialized
  Tensor::Tensor(const shape_t& shape) : shape_(shape), shape_prod_(calc_prod()), stride_(stride_t(shape.size())), offset_(0), storage_(Storage(numel_)) { init_set(); }
  
  // tensor with given shape, with data initialized to given value
  Tensor::Tensor(const shape_t& shape, dtype value) : shape_(shape), shape_prod_(calc_prod()), stride_(stride_t(shape.size())), offset_(0), storage_(Storage(numel_, value)) { init_set(); }
  
  // tensor with given shape, with data initialized by a generator function
  Tensor::Tensor(const shape_t& shape, std::function<dtype()> generator) : shape_(shape), shape_prod_(calc_prod()), stride_(stride_t(shape.size())), offset_(0), storage_(Storage(numel_, generator)) { init_set(); }
  
  // tensor with given shape, with a vector as underlying data
  Tensor::Tensor(const shape_t& shape, const vec<dtype>& data) : shape_(shape), shape_prod_(calc_prod()), stride_(stride_t(shape.size())), offset_(0), storage_(Storage(data)) {
    if (numel_ != data.size()) {
      throw std::runtime_error("invalid construct (Tensor set)");
    }
    init_set();
  }
  
  // tensor just change the view
  Tensor::Tensor(const shape_t& shape, int offset, Storage storage) : shape_(shape), shape_prod_(calc_prod()), stride_(stride_t(shape.size())), offset_(offset), storage_(storage) { init_set(); }
  
  // tensor constructed from metadata + storage
  Tensor::Tensor(const shape_t& shape, const stride_t& stride, int offset, Storage storage) : shape_(shape), shape_prod_(calc_prod()), stride_(stride), offset_(offset), storage_(storage) { init_copy(); }

  // shallow copy a tensor
  Tensor::Tensor(const Tensor& other) : Tensor(other.shape_, other.stride_, other.offset_, other.storage_) { }

  // shallow copy a tensor
  Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
      return *this;
    }
    shape_ = other.shape_;
    stride_ = other.stride_;
    offset_ = other.offset_;
    storage_ = other.storage_; // not clone
    numel_ = other.numel_;
    dim_ = other.dim_;
    is_contiguous_ = other.is_contiguous_;
    shape_prod_ = other.shape_prod_;
    return *this;
  }

  // only valid for singleton tensor
  Tensor& Tensor::operator=(dtype value) {
    if (numel_ != 1) {
      throw std::runtime_error("not singleton (Func Tensor::operator=)");
    }
    storage_[offset_] = value;
    return *this;
  }

  /* 
    destructor
  */
  Tensor::~Tensor() {}


  /*
    convert to dtype value
    only valid for singleton tensor
  */
  dtype Tensor::item() const {
    if (numel_ != 1) {
      throw std::runtime_error("not singleton (Func Tensor::item)");
    }
    return storage_[offset_];
  }

  /*
    utils
  */
 
  // convert python index to c++ index
  int pyindex(int index, int siz) {
    if (index < 0) index += siz;
    if (index < 0 || index >= siz) {
      throw std::runtime_error("out of range (Func pyindex)");
    }
    return index;
  }

  int Tensor::numel() const { return numel_; }

  int Tensor::dim() const { return dim_; }

  veci Tensor::size() const { return shape_; }

  int Tensor::size(int dim) const { return shape_[pyindex(dim, dim_)]; }

  bool Tensor::is_contiguous() const { return is_contiguous_; }


  /*
    clone, make contiguous, copy_ and scatter
  */
  
  bool match_shape(const shape_t& a, const shape_t b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (int i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) {
        return false;
      }
    }
    return true;
  }
  
  Tensor Tensor::clone() const {
    if (is_contiguous_) {
      return Tensor(shape_, stride_, offset_, storage_.clone());
    }
    Tensor new_tensor(shape_);
    for (int i = 0; i < numel_; i++) {
      new_tensor.data_at(i) = data_at(i);
    }
    return new_tensor;
  }

  Tensor Tensor::contiguous() const {
    if (is_contiguous_) {
      return *this;
    }
    return clone();
  }

  Tensor Tensor::copy_(const Tensor& other) const {
    if (this == &other) {
      return *this;
    }
    if (!match_shape(shape_, other.shape_)) {
      throw std::runtime_error("shape not match (Func Tensor::copy_)");
    }
    for (int i = 0; i < numel_; i++) {
      data_at(i) = other.data_at(i);
    }
    return *this;
  }

  Tensor Tensor::scatter_(int dim, const Tensor& index, const Tensor& src) const {
    dim = pyindex(dim, dim_);
    if (!match_shape(shape_, index.shape_) || !match_shape(shape_, src.shape_)) {
      throw std::runtime_error("shape not match (Func Tensor::scatter_)");
    }
    for (int i = 0; i < numel_; i++) {
      // not finish
    }
  }
  
  /*
    subscriptor
  */
 
  // convert python index to c++ index
  slice_t pyslice(slice_t sli, int siz) {
    if (sli.first < 0) sli.first += siz;
    if (sli.first < 0) sli.first = 0;
    if (sli.first >= siz) sli.first = siz;
    
    if (sli.second < 0) sli.second += siz;
    if (sli.second < 0) sli.second = 0;
    if (sli.second >= siz) sli.second = siz;
    
    if (sli.first >= sli.second) sli.first = sli.second = 0;
    
    return sli;
  }

  Tensor Tensor::operator[](const vec<slice_t>& slices) const {
    if (slices.size() > dim_) {
      throw std::runtime_error("invalid slice (Func Tensor::operator[])");
    }
    
    shape_t sli_shape(shape_);
    int sli_offset = offset_;
    
    for (int i = 0; i < slices.size(); i++) {
      auto [begin_, end_] = pyslice(slices[i], shape_[i]);
      sli_shape[i] = end_ - begin_;
      sli_offset += begin_ * stride_[i];
    }
    
    return Tensor(sli_shape, stride_, sli_offset, storage_);
  }

  Tensor Tensor::operator[](slice_t slice) const {
    if (dim_ == 0) {
      throw std::runtime_error("invalid slice (Func Tensor::operator[])");
    }
    
    shape_t sli_shape(shape_);
    int sli_offset = offset_;
    
    auto [begin_, end_] = pyslice(slice, shape_[0]);
    sli_shape[0] = end_ - begin_;
    sli_offset += begin_ * stride_[0];
    
    return Tensor(sli_shape, stride_, sli_offset, storage_);
  }

  Tensor Tensor::operator[](const veci& index) const {
    int idx_siz = index.size();
    if (idx_siz > dim_) {
      throw std::runtime_error("invalid index (Func Tensor::operator[])");
    }
    
    shape_t idx_shape(dim_ - idx_siz);
    stride_t idx_stride(dim_ - idx_siz);
    int idx_offset = offset_;
    
    for (int i = 0; i < idx_siz; i++) {
      int idx = pyindex(index[i], shape_[i]);
      idx_offset += idx * stride_[i];
    }
    
    for (int i = idx_siz; i < dim_; i++) {
      idx_shape[i - idx_siz] = shape_[i];
      idx_stride[i - idx_siz] = stride_[i];
    }
    
    return Tensor(idx_shape, idx_stride, idx_offset, storage_);
  }

  Tensor Tensor::operator[](int index) const {
    if (dim_ == 0) {
      throw std::runtime_error("invalid index (Func Tensor::operator[])");
    }
    
    shape_t idx_shape(dim_ - 1);
    stride_t idx_stride(dim_ - 1);
    int idx_offset = offset_;
    
    int idx = pyindex(index, shape_[0]);
    idx_offset += idx * stride_[0];
    
    for (int i = 1; i < dim_; i++) {
      idx_shape[i - 1] = shape_[i];
      idx_stride[i - 1] = stride_[i];
    }
    
    return Tensor(idx_shape, idx_stride, idx_offset, storage_);
  }

  /*
    operators
  */
  // General functions
  Tensor apply_unary_op(const Tensor& tes, std::function<dtype(dtype)> op) {
    vec<dtype> data(tes.numel());
    for (int i = 0; i < tes.numel(); i++) {
      data[i] = op(tes.data_at(i));
    }
    return Tensor(tes.get_shape(), data);
  }
  
  Tensor apply_binary_op(const Tensor& lhs, const Tensor& rhs, std::function<dtype(dtype, dtype)> op) {
    auto [LHS, RHS] = Tensor::broadcast(lhs, rhs);
    vec<dtype> data(LHS.numel());
    for (int i = 0; i < LHS.numel(); i++) {
      data[i] = op(LHS.data_at(i), RHS.data_at(i));
    }
    return Tensor(LHS.get_shape(), data);
  }
  
  // Unary negation
  Tensor Tensor::operator-() const {
    return apply_unary_op(*this, [](dtype a) -> dtype { return -a; });
  }
  
  // Element-wise addition
  Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype a, dtype b) -> dtype { return a + b; });
  }
  
  // Element-wise subtraction
  Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype a, dtype b) -> dtype { return a - b; });
  }
  
  // Element-wise multiplication
  Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype a, dtype b) -> dtype { return a * b; });
  }
  
  // Element-wise division
  Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype a, dtype b) -> dtype { return a / b; });
  }
  
  // Comparison
  Tensor operator==(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype a, dtype b) -> dtype { return a == b; });
  }

  Tensor operator!=(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype a, dtype b) -> dtype { return a != b; });
  }
  
  Tensor operator<(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype a, dtype b) -> dtype { return a < b; });
  }

  Tensor operator<=(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype a, dtype b) -> dtype { return a <= b; });
  }

  Tensor operator>=(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype a, dtype b) -> dtype { return a >= b; });
  }

  Tensor operator>(const Tensor& lhs, const Tensor& rhs) {
    return apply_binary_op(lhs, rhs, [](dtype a, dtype b) -> dtype { return a > b; });
  }

  /*
    matrix multiplication
  */
  Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    // wait for shape
  }
  
  // Equivalent to matmul
  Tensor operator^(const Tensor& lhs, const Tensor& rhs) { return matmul(lhs, rhs); }

  /*
    other mathematical operations
  */
  Tensor Tensor::sign() const {
    return apply_unary_op(*this, [](dtype a) -> dtype {
      if (a > 0) return 1.0;
      if (a < 0) return -1.0;
      return 0.0;
    });
  }

  Tensor Tensor::abs() const {
    return apply_unary_op(*this, [](dtype a) -> dtype { return std::abs(a); });
  }
  Tensor abs(const Tensor& tensor) { return tensor.abs(); }

  Tensor Tensor::sin() const {
    return apply_unary_op(*this, [](dtype a) -> dtype { return std::sin(a); });
  }
  Tensor sin(const Tensor& tensor) { return tensor.sin(); }

  Tensor Tensor::cos() const {
    return apply_unary_op(*this, [](dtype a) -> dtype { return std::cos(a); });
  }
  Tensor cos(const Tensor& tensor) { return tensor.cos(); }
  
  Tensor Tensor::tanh() const {
    return apply_unary_op(*this, [](dtype a) -> dtype { return std::tanh(a); });
  }
  Tensor tanh(const Tensor& tensor) { return tensor.tanh(); }

  Tensor Tensor::clamp(dtype min, dtype max) const {
    return apply_unary_op(*this, [min, max](dtype a) -> dtype { return std::min(max, std::max(min, a)); });
  }

  Tensor clamp(const Tensor& tensor, dtype min, dtype max) { return tensor.clamp(min, max); }

  Tensor Tensor::log() const {
    return apply_unary_op(*this, [](dtype a) -> dtype { return std::log(a); });
  }

  Tensor log(const Tensor& tensor) { return tensor.log(); }

  Tensor Tensor::exp() const {
    return apply_unary_op(*this, [](dtype a) -> dtype { return std::exp(a); });
  }

  Tensor exp(const Tensor& tensor) { return tensor.exp(); }

  Tensor Tensor::pow(dtype exponent) const {
    return apply_unary_op(*this, [exponent](dtype a) -> dtype { return std::pow(a, exponent); });
  }

  Tensor pow(const Tensor& tensor, dtype exponent) { return tensor.pow(exponent); }

  Tensor Tensor::sqrt() const {
    return apply_unary_op(*this, [](dtype a) -> dtype { return std::sqrt(a); });
  }

  Tensor sqrt(const Tensor& tensor) { return tensor.sqrt(); }

  Tensor Tensor::sum(int dim, bool keepdims) const {
    
  }

  Tensor sum(const Tensor& tensor, int dim, bool keepdims) { return tensor.sum(dim, keepdims); }

  std::pair<Tensor, Tensor> Tensor::max(int dim, bool keepdims) const {
    
  }

  std::pair<Tensor, Tensor> max(const Tensor& tensor, int dim, bool keepdims) { return tensor.max(dim, keepdims); }

  Tensor Tensor::softmax(int dim) const {
    
  }
  Tensor softmax(const Tensor& tensor, int dim) { return tensor.softmax(dim); }

  /*
    helper constructor
  */

  Tensor Tensor::ones_like() const { return ones(shape_); }
  Tensor Tensor::zeros_like() const { return zeros(shape_); }
  Tensor Tensor::randn_like() const { return randn(shape_); }
  Tensor Tensor::empty_like() const { return empty(shape_); }

  /*
    shape manipulation
  */
 
  shape_t deduce_shape(const shape_t& a, const int& numel_) {
    int prod = 1, cnt = 0, pos = -1;
    for (int i = 0; i < a.size(); i++) {
      if (a[i] == -1) {
        cnt++, pos = i;
      } else {
        prod *= a[i];
      }
    }
    if (cnt > 1 || cnt == 0 && prod != numel_ || cnt == 1 && (prod == 0 || numel_ % prod)) {
      throw std::runtime_error("invalid purposed shape (Func: deduce_shape)");
    }
    
    shape_t d_shape(a);
    if (cnt == 1) {
      d_shape[pos] = numel_ / prod;
    }
    return d_shape;
  }

  Tensor Tensor::permute(veci p) const {
    if (p.size() > dim_) {
      throw std::runtime_error("invalid size of p (Func Tensor::permute)");
    }
    for (int i = 0; i < p.size(); i++) {
      p[i] = pyindex(p[i], dim_);
    }
    
    veci sort_p(p);
    std::sort(sort_p.begin(), sort_p.end());
    
    for (int i = 1; i < p.size(); i++) {
      if (sort_p[i] == sort_p[i - 1]) {
        throw std::runtime_error("repeated elements of p (Func Tensor::permute)");
      }
    }
    
    shape_t p_shape(shape_);
    stride_t p_stride(stride_);
    for (int i = 0; i < p.size(); i++) {
      p_shape[sort_p[i]] = shape_[p[i]];
      p_stride[sort_p[i]] = stride_[p[i]];
    }
    
    return Tensor(p_shape, p_stride, offset_, storage_);
  }

  Tensor Tensor::transpose(int dim1, int dim2) const {
    dim1 = pyindex(dim1, dim_);
    dim2 = pyindex(dim2, dim_);
    if (dim1 == dim2) {
      return *this;
    }
    
    shape_t t_shape(shape_);
    stride_t t_stride(stride_);
    
    std::swap(t_shape[dim1], t_shape[dim2]);
    std::swap(t_stride[dim1], t_stride[dim2]);
    
    return Tensor(t_shape, t_stride,offset_, storage_);
  }

  Tensor Tensor::reshape(const shape_t& purposed_shape, bool copy) const {
    if (!is_contiguous_ || copy) {
      return clone().view(purposed_shape);
    }
    return view(purposed_shape);
  }

  Tensor Tensor::view(const shape_t& purposed_shape) const {
    if (!is_contiguous_) {
      throw std::runtime_error("viewing discontiguous tensor (Func Tensor::view)");
    }
    
    shape_t dd_shape(deduce_shape(purposed_shape, numel_));
    
    return Tensor(dd_shape, offset_, storage_);
  }

  Tensor Tensor::narrow(int dim, int start, int length, bool copy) const {
    dim = pyindex(dim, dim_);
    start = pyindex(start, shape_[dim]);
    
    if (start + length > shape_[dim]) {
      throw std::runtime_error("invalid length (Func: Tensor::narrow)");
    }
    
    shape_t nr_shape(shape_);
    nr_shape[dim] = length;
    int nr_offset = offset_ + start * stride_[dim];
    
    Tensor nr_tensor(nr_shape, stride_, nr_offset, storage_);
    
    return copy ? nr_tensor.clone() : nr_tensor;
  }

  vec<Tensor> Tensor::chunk(int chunks, int dim) const {
    if (chunks <= 0) {
      throw std::runtime_error("invalid number of chunks (Func: Tensor::chunk)");
    }
    dim = pyindex(dim, dim_);
    
    int split_size = (shape_[dim] - 1 + chunks) / chunks;
    
    return split(dim, split_size);
  }

  vec<Tensor> Tensor::split(int dim, int split_size) const {
    if (split_size <= 0) {
      throw std::runtime_error("invalid size (Func: Tensor::split)");
    }
    dim = pyindex(dim, dim_);
    
    int chunk = (shape_[dim] - 1 + split_size) / split_size;
    
    vec<Tensor> vect(chunk);
    for (int i = 0; i < chunk; i++) {
      int start = i * split_size;
      int len = std::min(shape_[dim] - start, split_size);
      vect[i] = narrow(dim, start, len);
    }
    
    return vect;
  }
  
  vec<Tensor> Tensor::split(int dim, veci split_sections) const {
    int sum = 0;
    for (int siz : split_sections) {
      if (siz <= 0) {
        throw std::runtime_error("invalid size (Func: Tensor::split)");
      }
      sum += siz;
    }
    dim = pyindex(dim, dim_);
    if (sum != shape_[dim]) {
      throw std::runtime_error("sum not match (Func: Tensor::split)");
    }
    
    vec<Tensor> vect(split_sections.size());
    int start = 0;
    for (int i = 0; i < split_sections.size(); i++) {
      vect[i] = narrow(dim, start, split_sections[i]);
      start += split_sections[i];
    }
    
    return vect;
  }
  
  Tensor Tensor::stack(const vec<Tensor>& inputs, int dim) {
    if (inputs.empty()) {
      throw std::runtime_error("empty tensorlist (Func: Tensor::stack)");
    }
    
    vec<Tensor> usq_inputs(inputs.size());
    for (int i = 0; i < inputs.size(); i++) {
      usq_inputs[i] = inputs[i].unsqueeze(dim);
    }
    
    return cat(usq_inputs, dim);
  }

  Tensor Tensor::cat(const vec<Tensor>& inputs, int dim) {
    if (inputs.empty()) {
      throw std::runtime_error("empty tensorlist (Func: Tensor::cat)");
    }
    
    shape_t cat_shape(inputs[0].shape_);
    int cat_dim = cat_shape.size();
    
    dim = pyindex(dim, cat_dim);
    
    for (int i = 1; i < inputs.size(); i++) {
      auto shp = inputs[i].shape_;
      
      if (shp.size() != cat_dim) {
        throw std::runtime_error("dimension should be same (Func: Tensor::cat)");
      }
      
      for (int i = 0; i < cat_dim; i++) {
        if (i == dim) continue;
        if (cat_shape[i] != shp[i]) {
          throw std::runtime_error("shape should be same (Func: Tensor::cat)");
        }
      }
      cat_shape[dim] += shp[dim];
    }
    
    Tensor cat_tensor(cat_shape);
    int cat_offset = 0;
    
    for (int i = 0; i < inputs.size(); i++) {
      int len = inputs[i].shape_[dim];
      cat_tensor.narrow(dim, cat_offset, len).copy_(inputs[i]);
      cat_offset += len;
    }
    
    return cat_tensor;
  }

  Tensor Tensor::squeeze(int dim) const {
    dim = pyindex(dim, dim_);
    if (shape_[dim] != 1) {
      throw std::runtime_error("shape of dim is not 1 (Func Tensor::squeeze)");
    }
    
    shape_t sq_shape(shape_);
    shape_t sq_stride(stride_);
    
    sq_shape.erase(sq_shape.begin() + dim);
    sq_stride.erase(sq_stride.begin() + dim);
    
    return Tensor(sq_shape, sq_stride, offset_, storage_);
  }

  Tensor Tensor::unsqueeze(int dim) const {
    dim = pyindex(dim, dim_ + 1); // [0, dim_]
    
    shape_t usq_shape(shape_);
    shape_t usq_stride(stride_);
    
    usq_shape.insert(usq_shape.begin() + dim, 1);
    int len = dim == dim_ ? 1 : stride_[dim] * shape_[dim];
    usq_stride.insert(usq_stride.begin() + dim, len);
    
    return Tensor(usq_shape, usq_stride, offset_, storage_);
  }
  
  // check & calculate broadcast shape
  shape_t broadcast_shape(const shape_t& a, const shape_t &b) {
    int dim_ = std::max(a.size(), b.size());
    shape_t shape_(dim_);
    for (int i = 0, asiz, bsiz; i < dim_; i++) {
      if (i >= a.size()) asiz = 1;
      else asiz = a[a.size() - i - 1];
      if (i >= b.size()) bsiz = 1;
      else bsiz = b[b.size() - i - 1];
      
      if (asiz != bsiz && asiz != 1 && bsiz != 1) {
        throw std::runtime_error("not broadcastable (Func broadcast_shape");
      }
      shape_[dim_ - i - 1] = std::max(asiz, bsiz);
    }
    return shape_;
  }
    
  Tensor Tensor::broadcast_to(const shape_t& shape) const {
    int to_dim = shape.size();
    if (to_dim < dim_) {
      throw std::runtime_error("invalid broadcast shape (Func Tensor::broadcast_to)");
    }
    
    stride_t to_stride(to_dim, 0);
    for (int i = 0; i < dim_; i++) {
      if (shape[dim_ - i - 1] == 0 || shape_[dim_ - i - 1] > 1 && shape_[dim_ - i - 1] != shape[to_dim - i - 1]) {
        throw std::runtime_error("invalid broadcast shape (Func Tensor::broadcast_to)");
      }
      if (shape_[dim_ - i - 1] == 1 && shape[to_dim - i - 1] == 1 || shape_[dim_ - i - 1] > 1) {
        to_stride[to_dim - i - 1] = stride_[stride_.size() - i - 1];
      }
    }
    return Tensor(shape, to_stride, offset_, storage_);
  }

  std::pair<Tensor, Tensor> Tensor::broadcast(const Tensor& lhs, const Tensor& rhs) {
    shape_t new_shape(broadcast_shape(lhs.shape_, rhs.shape_));
    Tensor LHS(lhs.broadcast_to(new_shape));
    Tensor RHS(rhs.broadcast_to(new_shape));
    return std::make_pair(LHS, RHS);
  }
  
  vec<Tensor> Tensor::broadcast(const vec<Tensor>& tensors) {
    
  }



  /*
    helper constructors
  */
  Tensor to_singleton_tensor(dtype value, int dim) {
    
  }

  Tensor ones(const shape_t& shape) {
    return Tensor(shape, 1.0);
  }
  Tensor ones_like(const Tensor& ref) {
    return ref.zeros_like();
  }

  Tensor zeros(const shape_t& shape) {
    return Tensor(shape, 0.0);
  }
  Tensor zeros_like(const Tensor& ref) {
    return ref.zeros_like();
  }

  Tensor randn(const shape_t& shape) {
    // wait for modifying
  }
  Tensor randn_like(const Tensor& ref) {
    return ref.randn_like();
  }

  Tensor empty(const shape_t& shape) {
    return Tensor(shape);
  }
  Tensor empty_like(const Tensor& ref) {
    return ref.empty_like();
  }

  Tensor arange(dtype start, dtype end, dtype step) {}

  Tensor range(dtype start, dtype end, dtype step) {}

  Tensor linspace(dtype start, dtype end, int num_steps) {}
  
  /*
    Week3 adds-on
  */
  Tensor Tensor::mean(int dim, bool keepdims) const {}

  Tensor Tensor::var(int dim, bool keepdims, bool unbiased) const {}
};
