#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/TensorUtils.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

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

void checkDim(
    CheckedFrom c,
    const Tensor& tensor,
    const char* name,
    int pos, // 1-indexed
    int64_t dim) {
  TORCH_CHECK(
      tensor.dim() == dim,
      "Expected ",
      dim,
      "-dimensional tensor, but got ",
      tensor.dim(),
      "-dimensional tensor for ",
      TensorGeometryArg(TensorArg({tensor, name, pos})),
      " (while checking arguments for ",
      c,
      ")");
}

void checkDim(CheckedFrom c, const TensorGeometryArg& t, int64_t dim) {
  TORCH_CHECK(t->dim() == dim,
    "Expected ", dim, "-dimensional tensor, but got ", t->dim(),
    "-dimensional tensor for ", t," (while checking arguments for ", c, ")");
}

void checkDimRange(CheckedFrom c, const TensorGeometryArg& t, int64_t dim_start, int64_t dim_end) {
  TORCH_CHECK(
    t->dim() >= dim_start && t->dim() < dim_end,
    "Expected ", dim_start, " to ", (dim_end - 1), " dimensions, but got ",
    t->dim(), "-dimensional tensor for ", t, " (while checking arguments for ",
    c, ")");
}

void checkContiguous(CheckedFrom c, const TensorGeometryArg& t) {
  TORCH_CHECK(
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
  TORCH_CHECK(
    t->sizes().equals(sizes),
    "Expected tensor of size ", sizes, ", but got tensor of size ", t->sizes(),
    " for ", t, " (while checking arguments for ", c, ")");
}

void checkSize_symint(CheckedFrom c, const TensorGeometryArg& t, c10::SymIntArrayRef sizes) {
  checkDim(c, t, sizes.size());
  TORCH_CHECK(
    t->sym_sizes().equals(sizes),
    "Expected tensor of size ", sizes, ", but got tensor of size ", t->sizes(),
    " for ", t, " (while checking arguments for ", c, ")");
}

void checkSize(CheckedFrom c, const TensorGeometryArg& t, int64_t dim, int64_t size) {
  TORCH_CHECK(
    t->size(dim) == size,
    "Expected tensor to have size ", size, " at dimension ", dim,
    ", but got size ", t->size(dim), " for ", t,
    " (while checking arguments for ", c, ")");
}

void checkSize_symint(CheckedFrom c, const TensorGeometryArg& t, int64_t dim, c10::SymInt size) {
  TORCH_CHECK(
    t->sym_size(dim) == size,
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
  TORCH_CHECK(
    t1->sizes().equals(t2->sizes()),
    "Expected tensor for ", t1, " to have same size as tensor for ", t2,
    "; but ", t1->sizes(), " does not equal ", t2->sizes(),
    " (while checking arguments for ", c, ")");
}

void checkAllSameSize(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameSize);
}

void checkNumel(CheckedFrom c, const TensorGeometryArg& t, int64_t numel) {
  TORCH_CHECK(
    t->numel() == numel,
    "Expected tensor for ", t, " to have ", numel,
    " elements; but it actually has ", t->numel(), " elements",
    " (while checking arguments for ", c, ")");
}

void checkSameNumel(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  TORCH_CHECK(
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
  if (t1->is_cpu() || t2->is_cpu()) {
    std::ostringstream oss;
    if (t1->is_cpu()) {
      oss << "Tensor for " << t1 << " is on CPU, ";
    }
    if (t2->is_cpu()) {
      oss << "Tensor for " << t2 << " is on CPU, ";
    }
    oss << "but expected " << ((!t1->is_cpu() && !t2->is_cpu()) ? "them" : "it")
        << " to be on GPU (while checking arguments for " << c << ")";
    AT_ERROR(oss.str());
  }
  TORCH_CHECK(
    t1->get_device() == t2->get_device(),
    "Expected tensor for ", t1, " to have the same device as tensor for ", t2,
    "; but device ", t1->get_device(), " does not equal ", t2->get_device(),
    " (while checking arguments for ", c, ")");
}

void checkAllSameGPU(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameGPU);
}

void checkSameType(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  TORCH_CHECK(
    t1->options().type_equal(t2->options()),
    "Expected tensor for ", t1, " to have the same type as tensor for ", t2,
    "; but type ", t1->toString(), " does not equal ", t2->toString(),
    " (while checking arguments for ", c, ")");
}

void checkScalarType(CheckedFrom c, const TensorArg& t, ScalarType ty) {
  TORCH_CHECK(
    t->scalar_type() == ty,
    "Expected tensor for ", t, " to have scalar type ", toString(ty),
    "; but got ", t->toString(), " instead (while checking arguments for ", c,
    ")");
}

void checkScalarTypes(CheckedFrom c, const TensorArg& t,
                      at::ArrayRef<ScalarType> l) {
    if (std::find(l.begin(), l.end(), t->scalar_type()) == l.end()) {
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
  TORCH_CHECK(
    t1->dim() == t2->dim(),
    "Expected tensor for ", t1, " to have the same dimension as tensor for ",
    t2, "; but ", t1->dim(), " does not equal ", t2->dim(),
    " (while checking arguments for ", c, ")");
}

void checkDefined(CheckedFrom c, const TensorArg& t) {
  TORCH_CHECK(
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
  TORCH_CHECK(
    !t.defined() || t.options().backend() == backend,
    "Expected tensor to have ", toString(backend),
    " Backend, but got tensor with ", toString(t.options().backend()), " Backend ",
    "(while checking arguments for ", c, ")");
}

void checkBackend(CheckedFrom c, at::ArrayRef<Tensor> tensors, at::Backend backend) {
  for (auto &t : tensors) {
    checkBackend(c, t, backend);
  }
}

void checkDeviceType(CheckedFrom c, const Tensor& t, DeviceType device_type) {
  TORCH_CHECK(
      !t.defined() || t.device().type() == device_type,
      "Expected tensor to have ", device_type,
      " DeviceType, but got tensor with ", t.device().type(), " DeviceType ",
      "(while checking arguments for ", c, ")");
}

void checkDeviceType(CheckedFrom c, at::ArrayRef<Tensor> tensors, at::DeviceType device_type) {
  for (auto &t : tensors) {
    checkDeviceType(c, t, device_type);
  }
}

void checkLayout(CheckedFrom c, const Tensor& t, Layout layout) {
  TORCH_CHECK(
    !t.defined() || t.layout() == layout,
    "Expected tensor to have ", layout,
    " Layout, but got tensor with ", t.layout(), " Layout ",
    "(while checking arguments for ", c, ")");
}

void checkLayout(CheckedFrom c, at::ArrayRef<Tensor> tensors, at::Layout layout) {
  for (auto &t : tensors) {
    checkLayout(c, t, layout);
  }
}

void * maybe_data_ptr(const Tensor& tensor) {
  return tensor.defined() ? (void *)tensor.data_ptr() : nullptr;
}

void * maybe_data_ptr(const TensorArg& tensor) {
  return tensor->defined() ? (void *)tensor->data_ptr() : nullptr;
}

void check_dim_size(
    const Tensor& tensor,
    int64_t dim,
    int64_t dim_size,
    int64_t size) {
  /* Check dimension size of a tensor */
  TORCH_CHECK(
      tensor.dim() == dim && tensor.size(dim_size) == size,
      "Expected a tensor of dimension ",
      dim,
      " and tensor.size[",
      dim_size,
      "] == ",
      size,
      " but got: dimension ",
      tensor.dim(),
      " and tensor.size[",
      dim_size,
      "] = ",
      tensor.size(dim_size));
}

namespace detail {

std::vector<int64_t> defaultStrides(IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for(size_t i = sizes.size(); i > 0; --i) {
    strides[i-1] = stride;
    stride *= sizes[i-1];
  }
  return strides;
}

// On a high level,
// 1. separate `oldshape` into chunks of dimensions, where the dimensions are
//    ``contiguous'' in each chunk, i.e., oldstride[i] = oldshape[i+1] *
//     oldstride[i+1]
// 2. `newshape` must be able to be separated into same number of chunks as
//    `oldshape` was separated into, where each chunk of newshape has matching
//    ``numel'', i.e., number of subspaces, as the corresponding chunk of
//    `oldshape`.
//
// templatized for DimVector and IntArrayRef use cases,
// see overloads of computeStride() below.
//
template <typename ResultVec, typename NewShapeVec, typename Numel>
inline c10::optional<ResultVec> computeStride_impl(
    const NewShapeVec& oldshape,
    const NewShapeVec& oldstride,
    const NewShapeVec& newshape,
    ResultVec toResult(const NewShapeVec&)
) {
  if (oldshape.empty()) {
    return ResultVec(newshape.size(), 1);
  }

  // NOTE: stride is arbitrary in the numel() == 0 case;
  // to match NumPy behavior we copy the strides if the size matches, otherwise
  // we use the stride as if it were computed via resize.
  // This could perhaps be combined with the below code, but the complexity
  // didn't seem worth it.
  const Numel numel = c10::multiply_integers(oldshape);
  if (numel == 0 && oldshape.equals(newshape)) {
    return toResult(oldstride);
  }

  ResultVec newstride(newshape.size());
  if (numel == 0) {
    for (int64_t view_d = newshape.size() - 1; view_d >= 0; view_d--) {
      if (view_d == (int64_t)(newshape.size() - 1)) {
        newstride[view_d] = 1;
      } else {
        newstride[view_d] =
          std::max<Numel>(newshape[view_d+1], Numel(1)) * newstride[view_d+1];
      }
    }
    return newstride;
  }

  int64_t view_d = (int64_t)newshape.size() - 1;
  // stride for each subspace in the chunk
  Numel chunk_base_stride = oldstride.back();
  // numel in current chunk
  Numel tensor_numel = 1;
  Numel view_numel = 1;
  for (int64_t tensor_d = oldshape.size() - 1; tensor_d >= 0; tensor_d--) {
    tensor_numel *= oldshape[tensor_d];
    // if end of tensor size chunk, check view
    if ((tensor_d == 0) ||
        (oldshape[tensor_d - 1] != 1 &&
         oldstride[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
      while (view_d >= 0 &&
            (view_numel < tensor_numel || newshape[view_d] == 1)) {
        newstride[view_d] = view_numel * chunk_base_stride;
        view_numel *= newshape[view_d];
        view_d--;
      }
      if (view_numel != tensor_numel) {
        return c10::nullopt;
      }
      if (tensor_d > 0) {
        chunk_base_stride = oldstride[tensor_d - 1];
        tensor_numel = 1;
        view_numel = 1;
      }
    }
  }
  if (view_d != -1) {
    return c10::nullopt;
  }
  return newstride;
}

c10::optional<std::vector<int64_t>> computeStride(
    IntArrayRef oldshape,
    IntArrayRef oldstride,
    IntArrayRef newshape) {
  auto toResult = [](const IntArrayRef& a) { return a.vec(); };
  return computeStride_impl<std::vector<int64_t>, IntArrayRef, int64_t>(oldshape, oldstride, newshape, toResult);
}

c10::optional<SymDimVector> computeStride(
    c10::SymIntArrayRef oldshape,
    c10::SymIntArrayRef oldstride,
    c10::SymIntArrayRef newshape) {
  auto toResult = [](const SymIntArrayRef& a) { return SymDimVector(a); };
  return computeStride_impl<SymDimVector, c10::SymIntArrayRef, c10::SymInt>(oldshape, oldstride, newshape, toResult);
}

c10::optional<DimVector> computeStride(
    IntArrayRef oldshape,
    IntArrayRef oldstride,
    const DimVector& newshape) {
  auto toResult = [](const IntArrayRef& a) { return DimVector(a); };
  return computeStride_impl<DimVector, IntArrayRef, int64_t>(oldshape, oldstride, newshape, toResult);
}

}  // namespace detail
}  // namespace at
