#include "ATen/Config.h"
#include "ATen/TensorUtils.h"

#include "ATen/ATen.h"

#include <ostream>
#include <sstream>

namespace at {

std::ostream& operator<<(std::ostream & out, TensorGeometryArg t) {
  if (t.pos == 0) {
    // 0 is distinguished; it usually indicates 'self' or the return
    // tensor
    out << "'" << t.name << "'";
  } else {
    out << "argument #" << t.pos << " '" << t.name << "'";
  }
  return out;
}

void checkDim(CheckedFrom c, const TensorGeometryArg& t, int64_t dim) {
  if (t->dim() != dim) {
    std::ostringstream oss;
    oss << "Expected " << dim << "-dimensional tensor, but got "
        << t->dim() << "-dimensional tensor for " << t
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkDimRange(CheckedFrom c, const TensorGeometryArg& t, int64_t dim_start, int64_t dim_end) {
  if (t->dim() < dim_start || t->dim() >= dim_end) {
    std::ostringstream oss;
    oss << "Expected " << dim_start << " to " << (dim_end - 1) << " dimensions, but got "
        << t->dim() << "-dimensional tensor for " << t
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkContiguous(CheckedFrom c, const TensorGeometryArg& t) {
  if (!t->is_contiguous()) {
    std::ostringstream oss;
    oss << "Expected contiguous tensor, but got non-contiguous tensor for " << t
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkAllContiguous(CheckedFrom c, at::ArrayRef<TensorArg> ts) {
  for (auto& t : ts) {
    if (!t->defined()) continue;
    checkContiguous(c, t);
  }
}

void checkSize(CheckedFrom c, const TensorGeometryArg& t, IntList sizes) {
  checkDim(c, t, sizes.size());
  if (!t->sizes().equals(sizes)) {
    std::ostringstream oss;
    oss << "Expected tensor of size " << sizes << ", but got tensor of size "
        << t->sizes() << " for " << t
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkSize(CheckedFrom c, const TensorGeometryArg& t, int64_t dim, int64_t size) {
  if (t->size(dim) != size) {
    std::ostringstream oss;
    oss << "Expected tensor to have size " << size << " at dimension " << dim
        << ", but got size " << t->size(dim) << " for " << t
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkAllSame(CheckedFrom c, ArrayRef<TensorArg> tensors, void(*fn)(CheckedFrom, const TensorArg&, const TensorArg&)) {
  const TensorArg* t0 = nullptr;
  for (auto& t : tensors) {
    if (!t->defined()) continue;
    if (t0 != nullptr) {
      fn(c, *t0, t);
    } else {
      t0 = &t;
    }
  }
}

void checkSameSize(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  if (!t1->sizes().equals(t2->sizes())) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t1 << " to have same size as tensor for "
        << t2 << "; but " << t1->sizes() << " does not equal " << t2->sizes()
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkAllSameSize(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameSize);
}

void checkNumel(CheckedFrom c, const TensorGeometryArg& t, int64_t numel) {
  if (t->numel() != numel) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t << " to have "
        << numel << " elements; but it actually has " << t->numel() << " elements"
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkSameNumel(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  if (t1->numel() != t2->numel()) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t1 << " to have same number of elements as tensor for "
        << t2 << "; but " << t1->numel() << " does not equal " << t2->numel()
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkAllSameNumel(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameNumel);
}

void checkSameGPU(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  if (t1->get_device() != t2->get_device()) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t1 << " to have the same device as "
        << "tensor for " << t2 << "; but device " << t1->get_device() << " "
        << "does not equal " << t2->get_device()
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkAllSameGPU(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameGPU);
}

void checkSameType(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  if (t1->type() != t2->type()) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t1 << " to have the same type as "
        << "tensor for " << t2 << "; but type " << t1->toString() << " "
        << "does not equal " << t2->toString()
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkScalarType(CheckedFrom c, const TensorArg& t, ScalarType ty) {
  if (t->type().scalarType() != ty) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t << " to have scalar type "
        << toString(ty) << "; but got " << t->toString()
        << " instead (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkScalarTypes(CheckedFrom c, const TensorArg& t,
                      at::ArrayRef<ScalarType> l) {
    if (std::find(l.begin(), l.end(), t->type().scalarType()) == l.end()) {
      std::ostringstream oss;
      oss << "Expected tensor for " << t << " to have one of the following "
          << "scalar types: ";
      size_t i = 0;
      for (auto ty : l) {
        if (i != 0) {
          oss << ", ";
        }
        oss << toString(ty);
        i++;
      }
      oss << "; but got " << t->toString()
          << " instead (while checking arguments for " << c << ")";
      throw std::runtime_error(oss.str());
    }
}

void checkAllSameType(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameType);
}

void checkSameDim(CheckedFrom c, const TensorGeometryArg& t1, const TensorGeometryArg& t2) {
  if (t1->dim() != t2->dim()) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t1 << " to have the same dimension as "
        << "tensor for " << t2 << "; but " << t1->dim() << " "
        << "does not equal " << t2->dim()
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkDefined(CheckedFrom c, const TensorArg& t) {
  if (!t->defined()) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t << " to be non-null, "
        << "but it was undefined "
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkAllDefined(CheckedFrom c, ArrayRef<TensorArg> ts) {
  // NB: don't filter defined here
  for (auto t : ts) {
    checkDefined(c, t);
  }
}

void checkBackend(CheckedFrom c, const Tensor& t, Backend backend) {
  if (t.type().backend() != backend) {
    std::ostringstream oss;
    oss << "Expected tensor to have " << toString(t.type().backend()) << " Backend, but got tensor with "
        << toString(t.type().backend()) << " Backend "
        << "(while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkBackend(CheckedFrom c, ArrayRef<Tensor> tensors, at::Backend backend) {
  for (auto &t : tensors) {
    checkBackend(c, t, backend);
  }
}

void * maybe_data_ptr(const Tensor& tensor) {
  return tensor.defined() ? (void *)tensor.data_ptr() : nullptr;
}

void * maybe_data_ptr(const TensorArg& tensor) {
  return tensor->defined() ? (void *)tensor->data_ptr() : nullptr;
}
}
