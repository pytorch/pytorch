#pragma once

#include <ATen/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/SparseTensorImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

namespace at {
// This struct defines the core implementation of the GCS 
// (Generalized Compresses Storage) sparse format. This
// format has been inspired from the paper 
// "Efficient storage scheme for n-dimensional sparse array: GCRS/GCCS"
// by Md Abu Hanif Shaikh and K.M. Azharul Hasan, published
// in 2015 (https://ieeexplore.ieee.org/document/7237032/). 
// In that paper the authors propose a compressed format
// for storing N-way sparse tensors which ends up exactly like
// the CSR format for 2-D tensors. We slightly tweak their
// approach to allow an arbitrary number of dimensions to
// from an N-D tensor be collapsed into two dimensions which
// can be represented using similar data structures as that 
// used in the CSR format. We call this new sparse format the
// "GCS" format.

// Since the GCS format allows the user to collapse contiguous
// dimensions of their choice into 2-D tensors, use of optimized
// routines meant for CSR formats from MKL or MAGMA can be
// used for performing operations on the collapsed dimensions.
// For further information on usage and best practices see
// the documentation of the torch.sparse_gcs_tensor() method
// and the benchmarks in the benchmarks/sparse folder for
// comparisons against COO.

// We use four Tensors for representing the GCS format:
// row_indices_, col_indices_, values_ and reduction_.
// The row_indices_, col_indices_ and values_ tensors
// are very similar to the tensors of the CSR format used
// for storing compressed row indices, col indices and
// non-zero values respectively. The reduction_ tensor
// is a new introduction into GCS that stores the dimensions
// of the tensor that must be collapsed 
struct TORCH_API SparseGCSTensorImpl : public TensorImpl {
  Tensor pointers_;
  Tensor indices_;
  Tensor values_;
  Tensor reduction_;

  // Data for making index conversion operations faster.

  // strides of the first half of the split dimensions.
  std::vector<int> strides0_;
  // strides of the second half of the split dimensions.
  std::vector<int> strides1_;
  // dims of the first half of the split dimensions.
  std::vector<int> dims0_;
  // dims of the second half of the split dimensions.
  std::vector<int> dims1_;
  // Dimension at which we split the tensor dimensions into two groups for reduction to
  // a 2D GCS tensor.
  int rsplit_dim_;           
 public:
  explicit SparseGCSTensorImpl(at::DispatchKeySet, const caffe2::TypeMeta&);

  void resize_and_clear_(int64_t nnz_size, int64_t ptr_size, int64_t redux_size, IntArrayRef size);
  void resize_as_(const Tensor& src);
  

  void set_member_tensors_unsafe(const Tensor& pointers, const Tensor& indices,
                                 const Tensor& values, const Tensor& reduction);

  std::vector<int> strides0() const { return strides0_; }
  std::vector<int> strides1() const { return strides1_; }
  std::vector<int> dims0() const { return dims0_; }
  std::vector<int> dims1() const { return dims1_; }
  int rsplit_dim() const { return rsplit_dim_; }
  
  Tensor pointers() const { return pointers_; }
  Tensor indices() const { return indices_; }
  Tensor values() const { return values_; }
  Tensor reduction() const { return reduction_; }
  int nnz() const { return values_.size(0); } // TODO: methods like these also exist in COO tensor. Deduplicate?

 private :
  
  explicit SparseGCSTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta& data_type,
                               at::Tensor pointers, at::Tensor indices, at::Tensor values, at::Tensor reduction);

  void make_strides(int shape_begin, std::vector<int>& strides, std::vector<int>& dims);
};
} // namespace at
