/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#ifndef _WIN32
#include <qnnpack/AlignedAllocator.h>
#endif
#include <pytorch_qnnpack.h>
#include <qnnpack/common.h>
#include <qnnpack/math.h>

#ifdef QNNPACK_BCSRMATRIX_DEBUG
#include <iostream>
#endif // QNNPACK_BCSRMATRIX_DEBUG

namespace qnnpack {

template <typename T>
struct OwnedOrBorrowedVector {
  using VECTOR_T =
#ifndef _WIN32
      std::vector<T, AlignedAllocator<T, 16>>;
#else
      std::vector<T>;
#endif

  // Only one of owned_vec_data_ or borrowed_tuple_data_ will be meaningfully
  // populated.
  // A union could potentially be used here to reduce memory usage.
  // std::variant is not used here because it causes internal build errors
  // due to incompatibility.
  VECTOR_T owned_vec_data_;
  std::tuple<T*, uint32_t> borrowed_tuple_data_;
  bool owned;

  VECTOR_T& vector() {
    assert(owned);
    return owned_vec_data_;
  }

  uint32_t size() const {
    if (owned) {
      return owned_vec_data_.size();
    } else {
      return std::get<1>(borrowed_tuple_data_);
    }
  }

  const T* data() const {
    if (owned) {
      return owned_vec_data_.data();
    } else {
      return std::get<0>(borrowed_tuple_data_);
    }
  }

  const T& operator[](int i) const {
    return data()[i];
  }

  OwnedOrBorrowedVector() : owned(true) {}

  OwnedOrBorrowedVector(T* data_ptr, const uint32_t size)
      : borrowed_tuple_data_(std::tuple<T*, uint32_t>(data_ptr, size)),
        owned(false) {}
};

struct BCSRMatrix {
  OwnedOrBorrowedVector<uint8_t> values;
  uint32_t col_block_size;  // input features block size
  uint32_t row_block_size;  // output features block size
  enum pytorch_qnnp_sparse_matrix_indices_dtype indices_dtype;
  virtual ~BCSRMatrix() = default;
  // Return void for the data ptrs because it doesn't require knowing the
  // underlying TypedBCSRMatrix indices dtype and that's how it's passed
  // into the qnnpack fully connected sparse op
  virtual const void* col_indices_data_ptr() const = 0;
  virtual const void* row_values_data_ptr() const = 0;
#ifdef QNNPACK_BCSRMATRIX_DEBUG
  virtual void print() const = 0;
#endif // QNNPACK_BCSRMATRIX_DEBUG
  /*
   * Unpack from BCSR to Dense
   * - Each value and zero point converted to int8_t by subtracting 128
   * - num_rows and num_cols are dimensions of dense weight tensor
   * - dst should be able to hold num_rows * num_cols elements
   * - zero_points should hold num_rows zero points
   */
  virtual void unpack(
      int8_t* dst,
      const int64_t num_rows,
      const int64_t num_cols,
      const uint8_t* zero_points) const = 0;
  virtual uint32_t max_index() const = 0;
};

template <typename INDICES_DTYPE>
struct TypedBCSRMatrix : BCSRMatrix {
  OwnedOrBorrowedVector<INDICES_DTYPE> col_indices;
  OwnedOrBorrowedVector<INDICES_DTYPE> row_values;
  TypedBCSRMatrix();
  const void* col_indices_data_ptr() const override;
  const void* row_values_data_ptr() const override;
#ifdef QNNPACK_BCSRMATRIX_DEBUG
  void print() const override;
#endif // QNNPACK_BCSRMATRIX_DEBUG
  void unpack(
      int8_t* dst,
      const int64_t num_rows,
      const int64_t num_cols,
      const uint8_t* zero_points) const override;
  uint32_t max_index() const override;

  ~TypedBCSRMatrix() override = default;
};

template <typename INDICES_DTYPE>
std::unique_ptr<BCSRMatrix> generateBlockCSRMatrix(
    const uint8_t* a,
    const size_t N,
    const size_t K,
    const uint32_t row_block_size,
    const uint32_t col_block_size,
    const uint8_t* zero_points) {
  assert(K > 0);
  std::unique_ptr<TypedBCSRMatrix<INDICES_DTYPE>> bcsr_mat =
      std::make_unique<TypedBCSRMatrix<INDICES_DTYPE>>();
  auto& row_values = bcsr_mat->row_values.vector();
  auto& col_indices = bcsr_mat->col_indices.vector();
  auto& values = bcsr_mat->values.vector();

  const uint32_t num_row_blocks = (N + row_block_size - 1) / row_block_size;
  // K must be > 0
  const uint32_t num_col_blocks = (K + col_block_size - 1) / col_block_size;

  row_values.reserve(num_row_blocks);
  uint32_t num_nnz_blocks{0};
  row_values.push_back(num_nnz_blocks);
  for (uint32_t i = 0; i < num_row_blocks; ++i) {
    for (uint32_t j = 0; j < num_col_blocks; ++j) {
      bool block_zero{true};
      for (uint32_t ib = 0; ib < row_block_size; ++ib) {
        uint32_t row_index = i * row_block_size + ib;
        if PYTORCH_QNNP_UNLIKELY(row_index >= N) {
          break;
        }
        for (uint32_t jb = 0; jb < col_block_size; ++jb) {
          uint32_t col_index = j * col_block_size + jb;
          if PYTORCH_QNNP_UNLIKELY(col_index >= K) {
            goto block_scanned;
          }
          if (*(a + row_index * K + col_index) != zero_points[row_index]) {
            block_zero = false;
            goto block_scanned;
          }
        }
      }
block_scanned:
      if (!block_zero) {
        col_indices.push_back(j);
        num_nnz_blocks++;
        for (uint32_t ib = 0; ib < row_block_size; ++ib) {
          uint32_t row_index = i * row_block_size + ib;
          if PYTORCH_QNNP_UNLIKELY(row_index >= N) {
            for (; row_index < (num_row_blocks * row_block_size); row_index++) {
              for (uint32_t jb = 0; jb < col_block_size; ++jb) {
                values.push_back(zero_points[N-1]);
              }
            }
            break;
          }
          for (uint32_t jb = 0; jb < col_block_size; ++jb) {
            uint32_t col_index = j * col_block_size + jb;
            if PYTORCH_QNNP_UNLIKELY(col_index >= K) {
              values.push_back(zero_points[row_index]);
            } else {
              uint8_t val = *(a + row_index * K + col_index);
              values.push_back(val);
            }
          }
        }
      }
    }
    row_values.push_back(num_nnz_blocks);
  }
  bcsr_mat->row_block_size = row_block_size;
  bcsr_mat->col_block_size = col_block_size;
  return bcsr_mat;
}

template <typename INDICES_DTYPE>
std::unique_ptr<BCSRMatrix> generateBlockCSRMatrix(
    INDICES_DTYPE* col_indices,
    INDICES_DTYPE* row_values,
    uint8_t* values,
    const int64_t col_indices_size,
    const int64_t row_values_size,
    const int64_t values_size,
    const int64_t row_block_size,
    const int64_t col_block_size) {
  std::unique_ptr<TypedBCSRMatrix<INDICES_DTYPE>> bcsr_mat =
      std::make_unique<TypedBCSRMatrix<INDICES_DTYPE>>();
  bcsr_mat->col_indices =
      OwnedOrBorrowedVector<INDICES_DTYPE>(col_indices, col_indices_size);
  bcsr_mat->row_values =
      OwnedOrBorrowedVector<INDICES_DTYPE>(row_values, row_values_size);
  bcsr_mat->values = OwnedOrBorrowedVector<uint8_t>(values, values_size);
  bcsr_mat->row_block_size = row_block_size;
  bcsr_mat->col_block_size = col_block_size;
  return bcsr_mat;
}

template <typename INDICES_DTYPE>
struct IndicesDtypeEnumTrait {
  static_assert(
      sizeof(INDICES_DTYPE) == 0,
      "Invalid dtype for IndicesDtypeEnumTrait");
};

template <>
struct IndicesDtypeEnumTrait<uint32_t> {
  const static pytorch_qnnp_sparse_matrix_indices_dtype dtype =
      pytorch_qnnp_sparse_matrix_indices_dtype_uint32_t;
};

template <>
struct IndicesDtypeEnumTrait<uint16_t> {
  const static pytorch_qnnp_sparse_matrix_indices_dtype dtype =
      pytorch_qnnp_sparse_matrix_indices_dtype_uint16_t;
};

template <>
struct IndicesDtypeEnumTrait<uint8_t> {
  const static pytorch_qnnp_sparse_matrix_indices_dtype dtype =
      pytorch_qnnp_sparse_matrix_indices_dtype_uint8_t;
};

template <typename INDICES_DTYPE>
TypedBCSRMatrix<INDICES_DTYPE>::TypedBCSRMatrix() {
  indices_dtype = IndicesDtypeEnumTrait<INDICES_DTYPE>::dtype;
}

template <typename INDICES_DTYPE>
const void* TypedBCSRMatrix<INDICES_DTYPE>::col_indices_data_ptr() const {
  return static_cast<const void*>(col_indices.data());
}

template <typename INDICES_DTYPE>
const void* TypedBCSRMatrix<INDICES_DTYPE>::row_values_data_ptr() const {
  return static_cast<const void*>(row_values.data());
}

#ifdef QNNPACK_BCSRMATRIX_DEBUG
template <typename INDICES_DTYPE>
void TypedBCSRMatrix<INDICES_DTYPE>::print() const {
  std::cout << "row block size:" << row_block_size << std::endl;
  std::cout << "col block size:" << col_block_size << std::endl;
  std::cout << "row ptr\n";
  std::cout
      << "indices dtype: uint"
      << static_cast<
             std::underlying_type_t<pytorch_qnnp_sparse_matrix_indices_dtype>>(
             indices_dtype)
      << "_t" << std::endl;
  for (uint32_t i = 0; i < row_values.size(); i++) {
    std::cout << (uint32_t)row_values[i] << ", ";
  }
  std::cout << std::endl;
  std::cout << "col indices\n";
  for (uint32_t i = 0; i < col_indices.size(); i++) {
    std::cout << (uint32_t)col_indices[i] << ", ";
  }
  std::cout << std::endl;
  std::cout << "Actual values\n";
  for (uint32_t i = 0; i < values.size(); i++) {
    std::cout << (uint32_t)values[i] << ", ";
  }
  std::cout << std::endl;
}
#endif // QNNPACK_BCSRMATRIX_DEBUG

template <typename INDICES_DTYPE>
void TypedBCSRMatrix<INDICES_DTYPE>::unpack(
    int8_t* dst,
    const int64_t num_rows,
    const int64_t num_cols,
    const uint8_t* zero_points) const {
  for (int64_t i = 0; i < num_rows; i++) {
    memset(
        dst + i * num_cols,
        static_cast<int8_t>(static_cast<int16_t>(zero_points[i]) - 128),
        num_cols * sizeof(int8_t));
  }

  const int64_t num_block_rows = static_cast<int64_t>(row_values.size()) - 1;
  const int64_t block_size = (int64_t)row_block_size * col_block_size;
  int64_t weight_values_num = 0;
  for (int64_t block_row_num = 0; block_row_num < num_block_rows;
       block_row_num++) {
    const int64_t num_blocks_in_current_block_row =
        row_values[block_row_num + 1] - row_values[block_row_num];
    for (int64_t k = 0; k < num_blocks_in_current_block_row;
         k++) { // iterate over each block in the row
      const int64_t block_start_row_num = block_row_num * row_block_size;
      const int64_t block_start_col_num =
          (int64_t)(col_indices[weight_values_num / block_size]) *
          col_block_size;
      for (int64_t l = 0; l < block_size;
           l++) { // iterate over each value in the block
        const int64_t row_num = block_start_row_num + l / col_block_size;
        const int64_t col_num = block_start_col_num + l % col_block_size;
        if (row_num < num_rows && col_num < num_cols) {
          dst[row_num * num_cols + col_num] = static_cast<int8_t>(
              static_cast<int16_t>(values[weight_values_num]) - 128);
        }
        weight_values_num++;
      }
    }
  }
}

template <typename INDICES_DTYPE>
uint32_t TypedBCSRMatrix<INDICES_DTYPE>::max_index() const {
  return static_cast<uint32_t>(std::max(
      *std::max_element(
          row_values.data(), row_values.data() + row_values.size()),
      *std::max_element(
          col_indices.data(), col_indices.data() + col_indices.size())));
}

/**
 * Given a BCSRMatrix (bcsr_) and a block of code enclosed in { }
 * (dispatch_body), run the block of code with the following in scope
 * 1) The BCSRMatrix's underlying TypedBCSRMatrix, called typed_bcsr
 * 2) The TypedBCSRMatrix's indices data type, called INDICES_DTYPE
 */
#define QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE(bcsr_, dispatch_body)        \
  [&bcsr = bcsr_]() {                                                          \
    switch (bcsr->indices_dtype) {                                             \
      case pytorch_qnnp_sparse_matrix_indices_dtype_uint32_t: {                \
        using INDICES_DTYPE = uint32_t;                                        \
        const qnnpack::TypedBCSRMatrix<INDICES_DTYPE>* typed_bcsr =            \
            static_cast<const qnnpack::TypedBCSRMatrix<INDICES_DTYPE>*>(       \
                bcsr.get());                                                   \
        return [&typed_bcsr]() dispatch_body();                                \
      }                                                                        \
      case pytorch_qnnp_sparse_matrix_indices_dtype_uint16_t: {                \
        using INDICES_DTYPE = uint16_t;                                        \
        const qnnpack::TypedBCSRMatrix<INDICES_DTYPE>* typed_bcsr =            \
            static_cast<const qnnpack::TypedBCSRMatrix<INDICES_DTYPE>*>(       \
                bcsr.get());                                                   \
        return [&typed_bcsr]() dispatch_body();                                \
      }                                                                        \
      case pytorch_qnnp_sparse_matrix_indices_dtype_uint8_t: {                 \
        using INDICES_DTYPE = uint8_t;                                         \
        const qnnpack::TypedBCSRMatrix<INDICES_DTYPE>* typed_bcsr =            \
            static_cast<const qnnpack::TypedBCSRMatrix<INDICES_DTYPE>*>(       \
                bcsr.get());                                                   \
        return [&typed_bcsr]() dispatch_body();                                \
      }                                                                        \
      case pytorch_qnnp_sparse_matrix_indices_dtype_invalid: {                 \
        assert(false);                                                         \
      }                                                                        \
    }                                                                          \
    /* Throw exception to avoid the following errors: */                       \
    /* - "non-void lambda does not return a value in all control paths" */     \
    /* - "control reaches end of non-void function" */                         \
    /* Throwing exception from within invalid case alone does not fix these */ \
    throw std::invalid_argument(                                               \
        "Invalid indices dtype in QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE"); \
  }()

} // namespace qnnpack
