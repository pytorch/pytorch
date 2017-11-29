#include "ATen/Check.h"

#include "ATen/ATen.h"

namespace at {

namespace {
  std::vector<TensorArg> filterDefined(ArrayRef<TensorArg> ts) {
    std::vector<TensorArg> r;
    r.reserve(ts.size());
    for (auto t : ts) {
      if (t->defined()) {
        r.push_back(t);
      }
    }
    return r;
  }
}

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

void checkDim(CheckedFrom c, TensorGeometryArg t, int64_t dim) {
  if (t->dim() != dim) {
    std::ostringstream oss;
    oss << "Expected " << dim << "-dimensional tensor, but got "
        << t->dim() << "-dimensional tensor for " << t
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkDimRange(CheckedFrom c, TensorGeometryArg t, int64_t dim_start, int64_t dim_end) {
  if (t->dim() < dim_start || t->dim() >= dim_end) {
    std::ostringstream oss;
    oss << "Expected " << dim_start << " to " << (dim_end - 1) << " dimensions, but got "
        << t->dim() << "-dimensional tensor for " << t
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkContiguous_(CheckedFrom c, TensorGeometryArg t) {
  if (!t->is_contiguous()) {
    std::ostringstream oss;
    oss << "Expected contiguous tensor, but got non-contiguous tensor for " << t
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkContiguous(CheckedFrom c, at::ArrayRef<TensorArg> ts) {
  for (auto t : filterDefined(ts)) {
    checkContiguous_(c, t);
  }
}

void checkSize(CheckedFrom c, TensorGeometryArg t, IntList sizes) {
  checkDim(c, t, sizes.size());
  if (!t->sizes().equals(sizes)) {
    std::ostringstream oss;
    oss << "Expected tensor of size " << sizes << ", but got tensor of size "
        << t->sizes() << " for " << t
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkSize(CheckedFrom c, TensorGeometryArg t, int64_t dim, int64_t size) {
  if (t->size(dim) != size) {
    std::ostringstream oss;
    oss << "Expected tensor to have size " << size << " at dimension " << dim
        << ", but got size " << t->size(dim) << " for " << t
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkSameSize(CheckedFrom c, TensorGeometryArg t1, TensorGeometryArg t2) {
  if (!t1->sizes().equals(t2->sizes())) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t1 << " to have same size as tensor for "
        << t2 << "; but " << t1->sizes() << " does not equal " << t2->sizes()
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkSameSize(CheckedFrom c, ArrayRef<TensorArg> tensors_) {
  auto tensors = filterDefined(tensors_);
  if (tensors.empty()) return;
  auto t0 = tensors.front();
  for (auto t : ArrayRef<TensorArg>{tensors}.slice(1)) {
    checkSameSize(c, t0, t);
  }
}

void checkNumel(CheckedFrom c, TensorGeometryArg t, int64_t numel) {
  if (t->numel() != numel) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t << " to have "
        << numel << " elements; but it actually has " << t->numel() << " elements"
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkSameNumel(CheckedFrom c, TensorGeometryArg t1, TensorGeometryArg t2) {
  if (t1->numel() != t2->numel()) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t1 << " to have same number of elements as tensor for "
        << t2 << "; but " << t1->numel() << " does not equal " << t2->numel()
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkSameNumel(CheckedFrom c, ArrayRef<TensorArg> tensors_) {
  auto tensors = filterDefined(tensors_);
  if (tensors.empty()) return;
  auto t0 = tensors.front();
  for (auto t : ArrayRef<TensorArg>{tensors}.slice(1)) {
    checkSameNumel(c, t0, t);
  }
}

void checkSameGPU(CheckedFrom c, TensorArg t1, TensorArg t2) {
  if (t1->get_device() != t2->get_device()) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t1 << " to have the same device as "
        << "tensor for " << t2 << "; but device " << t1->get_device() << " "
        << "does not equal " << t2->get_device()
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkSameGPU(CheckedFrom c, ArrayRef<TensorArg> tensors_) {
  auto tensors = filterDefined(tensors_);
  if (tensors.empty()) return;
  auto t0 = tensors.front();
  for (auto t : ArrayRef<TensorArg>{tensors}.slice(1)) {
    checkSameGPU(c, t0, t);
  }
}

void checkSameType(CheckedFrom c, TensorArg t1, TensorArg t2) {
  if (t1->type().ID() != t2->type().ID()) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t1 << " to have the same type as "
        << "tensor for " << t2 << "; but type " << t1->toString() << " "
        << "does not equal " << t2->toString()
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkScalarType(CheckedFrom c, TensorArg t, ScalarType ty) {
  if (t->type().scalarType() != ty) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t << " to have scalar type "
        << toString(ty) << "; but got " << t->toString()
        << " instead (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkSameType(CheckedFrom c, ArrayRef<TensorArg> tensors_) {
  auto tensors = filterDefined(tensors_);
  if (tensors.empty()) return;
  auto t0 = tensors.front();
  for (auto t : ArrayRef<TensorArg>{tensors}.slice(1)) {
    checkSameType(c, t0, t);
  }
}

void checkSameDim(CheckedFrom c, TensorGeometryArg t1, TensorGeometryArg t2) {
  if (t1->dim() != t2->dim()) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t1 << " to have the same dimension as "
        << "tensor for " << t2 << "; but " << t1->dim() << " "
        << "does not equal " << t2->dim()
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkDefined_(CheckedFrom c, TensorArg t) {
  if (!t->defined()) {
    std::ostringstream oss;
    oss << "Expected tensor for " << t << " to be non-null, "
        << "but it was undefined "
        << " (while checking arguments for " << c << ")";
    throw std::runtime_error(oss.str());
  }
}

void checkDefined(CheckedFrom c, ArrayRef<TensorArg> ts) {
  // NB: don't filter defined here
  for (auto t : ts) {
    checkDefined_(c, t);
  }
}

}
