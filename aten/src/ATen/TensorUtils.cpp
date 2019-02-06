#include <ATen/Config.h>
#include <ATen/TensorUtils.h>

#include <ATen/ATen.h>

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
  AT_CHECK(t->dim() == dim,
    "Expected ", dim, "-dimensional tensor, but got ", t->dim(),
    "-dimensional tensor for ", t," (while checking arguments for ", c, ")");
}

void checkDimRange(CheckedFrom c, const TensorGeometryArg& t, int64_t dim_start, int64_t dim_end) {
  AT_CHECK(
    t->dim() >= dim_start && t->dim() < dim_end,
    "Expected ", dim_start, " to ", (dim_end - 1), " dimensions, but got ",
    t->dim(), "-dimensional tensor for ", t, " (while checking arguments for ",
    c, ")");
}

void checkContiguous(CheckedFrom c, const TensorGeometryArg& t) {
  AT_CHECK(
    t->is_contiguous(),
    "Expected contiguous tensor, but got non-contiguous tensor for ", t,
     " (while checking arguments for ", c, ")");
}

void checkAllContiguous(CheckedFrom c, at::ArrayRef<TensorArg> ts) {
  for (auto& t : ts) {
    if (!t->defined()) continue;
    checkContiguous(c, t);
  }
}

void checkSize(CheckedFrom c, const TensorGeometryArg& t, IntArrayRef sizes) {
  checkDim(c, t, sizes.size());
  AT_CHECK(
    t->sizes().equals(sizes),
    "Expected tensor of size ", sizes, ", but got tensor of size ", t->sizes(),
    " for ", t, " (while checking arguments for ", c, ")");
}

void checkSize(CheckedFrom c, const TensorGeometryArg& t, int64_t dim, int64_t size) {
  AT_CHECK(
    t->size(dim) == size,
    "Expected tensor to have size ", size, " at dimension ", dim,
    ", but got size ", t->size(dim), " for ", t,
    " (while checking arguments for ", c, ")");
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
  AT_CHECK(
    t1->sizes().equals(t2->sizes()),
    "Expected tensor for ", t1, " to have same size as tensor for ", t2,
    "; but ", t1->sizes(), " does not equal ", t2->sizes(),
    " (while checking arguments for ", c, ")");
}

void checkAllSameSize(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameSize);
}

void checkNumel(CheckedFrom c, const TensorGeometryArg& t, int64_t numel) {
  AT_CHECK(
    t->numel() == numel,
    "Expected tensor for ", t, " to have ", numel,
    " elements; but it actually has ", t->numel(), " elements",
    " (while checking arguments for ", c, ")");
}

void checkSameNumel(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  AT_CHECK(
    t1->numel() == t2->numel(),
    "Expected tensor for ", t1,
    " to have same number of elements as tensor for ", t2, "; but ",
    t1->numel(), " does not equal ", t2->numel(),
    " (while checking arguments for ", c, ")");
}

void checkAllSameNumel(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameNumel);
}

void checkSameGPU(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  if (! (t1->is_cuda()) || ! (t2->is_cuda())) {
    std::ostringstream oss;
    if (! t1->is_cuda()) {
      oss << "Tensor for " << t1 << " is on CPU, ";
    }
    if (! t2->is_cuda()) {
      oss << "Tensor for " << t2 << " is on CPU, ";
    }
    oss << "but expected " << ((!(t1->is_cuda() || t2->is_cuda())) ? "them" : "it")
        << " to be on GPU (while checking arguments for " << c << ")";
    AT_ERROR(oss.str());
  }
  AT_CHECK(
    t1->get_device() == t2->get_device(),
    "Expected tensor for ", t1, " to have the same device as tensor for ", t2,
    "; but device ", t1->get_device(), " does not equal ", t2->get_device(),
    " (while checking arguments for ", c, ")");
}

void checkAllSameGPU(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameGPU);
}

void checkSameType(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  AT_CHECK(
    t1->type() == t2->type(),
    "Expected tensor for ", t1, " to have the same type as tensor for ", t2,
    "; but type ", t1->toString(), " does not equal ", t2->toString(),
    " (while checking arguments for ", c, ")");
}

void checkScalarType(CheckedFrom c, const TensorArg& t, ScalarType ty) {
  AT_CHECK(
    t->type().scalarType() == ty,
    "Expected tensor for ", t, " to have scalar type ", toString(ty),
    "; but got ", t->toString(), " instead (while checking arguments for ", c,
    ")");
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
      AT_ERROR(oss.str());
    }
}

void checkAllSameType(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameType);
}

void checkSameDim(CheckedFrom c, const TensorGeometryArg& t1, const TensorGeometryArg& t2) {
  AT_CHECK(
    t1->dim() == t2->dim(),
    "Expected tensor for ", t1, " to have the same dimension as tensor for ",
    t2, "; but ", t1->dim(), " does not equal ", t2->dim(),
    " (while checking arguments for ", c, ")");
}

void checkDefined(CheckedFrom c, const TensorArg& t) {
  AT_CHECK(
    t->defined(),
    "Expected tensor for ", t, " to be non-null, but it was undefined ",
    " (while checking arguments for ", c, ")");
}

void checkAllDefined(CheckedFrom c, ArrayRef<TensorArg> ts) {
  // NB: don't filter defined here
  for (auto t : ts) {
    checkDefined(c, t);
  }
}

void checkBackend(CheckedFrom c, const Tensor& t, Backend backend) {
  AT_CHECK(
    !t.defined() || t.type().backend() == backend,
    "Expected tensor to have ", toString(backend),
    " Backend, but got tensor with ", toString(t.type().backend()), " Backend ",
    "(while checking arguments for ", c, ")");
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

// See TensorUtils.h on why this is useful now that we cache is_contiguous.
bool geometry_is_contiguous(IntArrayRef sizes, IntArrayRef strides) {
  int64_t dim = sizes.size();
  int64_t expected_stride = 1;
  bool contig_if_nonempty = true;
  for (int64_t i = dim - 1; i >= 0; i--) {
    if (sizes[i] == 0) {
      return true;
    }
    if (contig_if_nonempty) {
      if (sizes[i] != 1 && strides[i] != expected_stride) {
        contig_if_nonempty = false;
      }
      expected_stride *= sizes[i];
    }
  }
  return contig_if_nonempty;
}

}
