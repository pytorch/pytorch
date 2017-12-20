#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/PinnedMemoryAllocator.h"
#include "ATen/WrapDimUtils.h"
#include <functional>
#include <numeric>
#include <type_traits>

namespace at {
namespace native {

Tensor& bernoulli_(Tensor& self, const Tensor& p, Generator* generator) {
  self.copy_(at::bernoulli(std::get<0>(expand_inplace(self, p)), generator));
  return self;
}

Tensor& bernoulli_(Tensor& self, double p, Generator* generator) {
  Tensor probs = self.type().toScalarType(kDouble).tensor({}).fill_(p);
  return native::bernoulli_(self, probs, generator);
}

Tensor type_as(const Tensor& self, const Tensor& other) {
  return self.toType(other.type());
}

Tensor expand_as(const Tensor& self, const Tensor& other) {
  return self.expand(other.sizes());
}

std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim) {
  int64_t dim_size = self.size(dim);
  int64_t num_splits = (dim_size + split_size - 1) / split_size;
  std::vector<Tensor> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - dim_size);

  for (int64_t i = 0; i < num_splits; ++i) {
    auto length = i < num_splits - 1 ? split_size : last_split_size;
    splits[i] = self.narrow(dim, i * split_size, length);
  }
  return splits;
}

Tensor slice(const Tensor& self, int64_t start, int64_t end, int64_t step, int64_t dim) {
  int64_t ndim = self.dim();
  AT_ASSERT(ndim > 0, "slice() cannot be applied to a 0-dim tensor.");
  dim = maybe_wrap_dim(dim, ndim);
  auto sizes = std::vector<int64_t>(self.sizes());
  auto strides = std::vector<int64_t>(self.strides());
  if (step <= 0) {
    // TODO: support negative strides
    throw std::runtime_error("slice step must be positive");
  }
  if (start < 0) {
    start += sizes[dim];
  }
  if (end < 0) {
    end += sizes[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= sizes[dim]) {
    start = sizes[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= sizes[dim]) {
    end = sizes[dim];
  }
  auto storage_offset = self.storage_offset() + start * strides[dim];
  auto len = end - start;
  sizes[dim] = (len + step - 1) / step;  // round-up
  strides[dim] *= step;
  return self.as_strided(sizes, strides, storage_offset);
}

Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  AT_ASSERT(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  auto cur_size = self.size(dim);
  if (start < 0 || start >= cur_size) {
    runtime_error("start out of range");
  }
  if (length <= 0 || start > cur_size - length) {
    runtime_error("length out of range");
  }
  return at::native::slice(self, start, start + length, 1, dim);
}

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
  int64_t ndim = self.dim();
  AT_ASSERT(ndim > 0, "select() cannot be applied to a 0-dim tensor.");
  dim = maybe_wrap_dim(dim, ndim);
  auto size = self.size(dim);
  if (index < -size || index >= size) {
    std::stringstream ss;
    ss << "select(): index " << index << " out of range for tensor of size ";
    ss << self.sizes() << " at dimension " << dim;
    throw std::runtime_error(ss.str());
  }
  if (index < 0) {
    index += size;
  }
  auto sizes = std::vector<int64_t>(self.sizes());
  auto strides = std::vector<int64_t>(self.strides());
  auto storage_offset = self.storage_offset() + index * strides[dim];
  sizes.erase(sizes.begin() + dim);
  strides.erase(strides.begin() + dim);
  return self.as_strided(sizes, strides, storage_offset);
}

std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  int64_t split_size = (self.size(dim) + chunks - 1) / chunks;
  // ensure this is dispatched through Tensor/Type, rather than the native function directly.
  return self.split(split_size, dim);
}

int64_t size(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());
  // wrap_dim guarantees bounds are correct.
  return self.sizes()[dim];
}

int64_t stride(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());
  // wrap_dim guarantees bounds are correct.
  return self.strides()[dim];
}

bool is_nonzero(const Tensor& self) {
  if (self.numel() != 1) {
    runtime_error("bool value of Tensor with more than one value is ambiguous");
  }
  Scalar localScalar = self.pImpl->localScalar();
  if (localScalar.isFloatingPoint()) {
    return localScalar.to<double>() != 0;
  } else if (localScalar.isIntegral()){
    return localScalar.to<int64_t>() != 0;
  }
  runtime_error("expected non-Tensor backed scalar");
}

bool is_same_size(const Tensor& self, const Tensor& other) {
  return self.sizes().equals(other.sizes());
}

bool is_cuda(const Tensor& self) {
  return self.type().is_cuda();
}

bool is_distributed(const Tensor& self) {
  return self.type().is_distributed();
}

bool is_sparse(const Tensor& self) {
  return self.type().is_sparse();
}

template <typename scalar>
struct IsSigned {
  static bool apply() { return std::is_signed<scalar>(); }
};

template<>
struct IsSigned<Half> {
  static bool apply() { return true; }
};

bool is_signed(const Tensor &self) {
  return dispatch_all<IsSigned>(self.type(), "is_signed");
}

Tensor permute(const Tensor& self, IntList dims) {
  auto nDims = self.dim();
  if (dims.size() != (size_t)nDims) {
    runtime_error("number of dims don't match in permute");
  }
  auto oldSizes = self.sizes();
  auto oldStrides = self.strides();
  std::vector<int64_t> newSizes(nDims);
  std::vector<int64_t> newStrides(nDims);
  std::vector<bool> seen(nDims);
  for (int64_t i = 0; i < nDims; i++) {
    auto dim = maybe_wrap_dim(dims[i], nDims);
    if (seen[dim]) {
      runtime_error("repeated dim in permute");
    }
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    newStrides[i] = oldStrides[dim];
  }
  return self.as_strided(newSizes, newStrides);
}

Tensor expand(const Tensor& self, IntList size) {
  if (size.size() < (size_t)self.dim()) {
    std::ostringstream ss;
    ss << "expand(" << self.type() << "{" << self.sizes() << "}, size=" << size
       << "): the number of sizes provided (" << size.size() << ") "
       << "must be greater or equal to the number of dimensions in the tensor ("
       << self.dim() << ")";
    throw std::runtime_error(ss.str());
  }

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) = inferExpandGeometry(self, size);

  return self.as_strided(expandedSizes, expandedStrides);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferSqueezeGeometry(const Tensor &tensor) {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  for(int64_t d = 0; d < tensor.dim(); d++) {
    if(tensor.sizes()[d] != 1) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }

  return std::make_tuple(sizes, strides);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferSqueezeGeometry(const Tensor& tensor, int64_t dim) {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  for(int64_t d = 0; d < tensor.dim(); d++) {
    if(d != dim || tensor.sizes()[dim] != 1) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }
  return std::make_tuple(sizes, strides);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferUnsqueezeGeometry(const Tensor& tensor, int64_t dim) {
  if (tensor.numel() == 0) {
    throw std::runtime_error("cannot unsqueeze empty tensor");
  }

  std::vector<int64_t> sizes(tensor.sizes());
  std::vector<int64_t> strides(tensor.strides());
  int64_t new_stride = dim >= tensor.dim() - 1 ? 1 : sizes[dim] * strides[dim];
  sizes.insert(sizes.begin() + dim, 1);
  strides.insert(strides.begin() + dim, new_stride);

  return std::make_tuple(sizes, strides);
}

Tensor squeeze(const Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

Tensor squeeze(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());

  if (self.sizes()[dim] != 1) {
    return self.as_strided(self.sizes().vec(), self.strides().vec());
  }
  auto g = inferSqueezeGeometry(self, dim);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

Tensor & squeeze_(Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

Tensor & squeeze_(Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());

  if (self.sizes()[dim] != 1) {
    return self.as_strided_(self.sizes().vec(), self.strides().vec());
  }
  auto g = inferSqueezeGeometry(self, dim);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

Tensor unsqueeze(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  auto g = inferUnsqueezeGeometry(self, dim);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

Tensor & unsqueeze_(Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  auto g = inferUnsqueezeGeometry(self, dim);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

// For backward, we save svd.
// http://www.ics.forth.gr/cvrl/publications/conferences/2000_eccv_SVD_jacobian.pdf
// But instead of gesvd SVD A = U(A) Sig(A) V(A)^T, which doesn't specify signs
// of determinants of U and V, we consider det(A) = \prod Sig_(A), where
//   1. A = U_(A) Sig_(A) V(A)^T
//   2. Sig_(A) and U_(A) can be different in signs in first row/col from
//      their counterparts so that U_(A) * V_(A) have +1 determinant
std::tuple<Tensor, Tensor, Tensor, Tensor> _det_with_svd(const Tensor& self) {
  if (!at::isFloatingType(self.type().scalarType()) ||
      self.dim() != 2 || self.size(0) != self.size(1)) {
    std::ostringstream ss;
    ss << "det(" << self.type() << "{" << self.sizes() << "}): expected a 2D "
       << "square tensor of floating types";
    throw std::runtime_error(ss.str());
  }
  // check symmetric
  bool symmetric = self.equal(self.transpose(0, 1));

  auto svd = self.svd(true);
  auto sigma = std::get<1>(svd);
  auto u = std::get<0>(svd);
  auto v = std::get<2>(svd);
  auto det = sigma.prod();
  if (!symmetric) {
    auto qr = self.geqrf();
    auto a = std::get<0>(qr);
    auto tau = std::get<1>(qr);
    // non-zero values in tau represent Householder reflectors, which has -1 det
    int64_t num_reflectors = tau.nonzero().size(0);
    auto qr_det = a.diag().prod();
    if (num_reflectors % 2 == 1) {
      qr_det = -qr_det;
    }
    det = qr_det;  // QR is more stable than svd, so use it anyways
    if ((qr_det < 0).any() ^ (det < 0).any()) {  // if different sign
      u.narrow(1, 0, 1).mul_(-1);
      sigma.narrow(0, 0, 1).mul_(-1);
    }
  }
  return std::make_tuple(det, u, sigma, v);
}

Tensor det(const Tensor& self) {
  return std::get<0>(self._det_with_svd());
}

Tensor stack(TensorList tensors, int64_t dim) {
  if (tensors.size() == 0) {
    throw std::runtime_error("stack expects a non-empty TensorList");
  }
  dim = maybe_wrap_dim(dim, tensors[0].dim() + 1);

  std::vector<Tensor> inputs(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    inputs[i] = tensors[i].unsqueeze(dim);
  }
  return at::cat(inputs, dim);
}

Tensor pin_memory(const Tensor& self) {
  if (self.type().backend() != kCPU) {
    runtime_error("cannot pin '%s' only CPU memory can be pinned", self.type().toString());
  }
  auto allocator = std::unique_ptr<Allocator>(new PinnedMemoryAllocator());
  auto tensor = self.type().tensorWithAllocator(self.sizes(), self.strides(), std::move(allocator));
  tensor.copy_(self);
  return tensor;
}

static Tensor maybeSqueeze(const Tensor & tensor, int64_t dim_tensor1, int64_t dim_tensor2) {
  if (dim_tensor1 == 1) {
    return tensor.squeeze(-2);
  } else if (dim_tensor2 == 1) {
    return tensor.squeeze(-1);
  } else {
    return tensor;
  }
}

/*
Matrix product of two Tensors.
The behavior depends on the dimensionality of the Tensors as follows:
- If both Tensors are 1-dimensional, the dot product (scalar) is returned.
- If both arguments are 2-dimensional, the matrix-matrix product is returned.
- If the first argument is 1-dimensional and the second argument is 2-dimensional,
  a 1 is prepended to its dimension for the purpose of the matrix multiply.
  After the matrix multiply, the prepended dimension is removed.
- If the first argument is 2-dimensional and the second argument is 1-dimensional,
  the matrix-vector product is returned.
- If both arguments are at least 1-dimensional and at least one argument is
  N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
  argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
  batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
  1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
  The non-matrix (i.e. batch) dimensions are broadcasted (and thus
  must be broadcastable).  For example, if tensor1 is a (j x 1 x n x m) Tensor
  and tensor2 is a (k x m x p) Tensor, the returned tensor will be an (j x k x n x p) Tensor.
*/
Tensor matmul(const Tensor & tensor1, const Tensor & tensor2) {
  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    return tensor1.dot(tensor2);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    return tensor1.mv(tensor2);
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return tensor1.mm(tensor2);
  } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
    // optimization: use mm instead of bmm by folding tensor1's batch into
    // its leading matrix dimension.

    Tensor t2 = dim_tensor2 == 1 ? tensor2.unsqueeze(-1) : tensor2;
    auto size1 = tensor1.sizes();
    auto size2 = t2.sizes();
    std::vector<int64_t> output_size;
    output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
    output_size.insert(output_size.end(), size2.end() - 1, size2.end());

    // fold the batch into the first dimension
    Tensor t1 = tensor1.contiguous().view({-1, size1[size1.size() - 1]});

    auto output = t1.mm(t2).view(output_size);
    if (dim_tensor2 == 1) {
      output = output.squeeze(-1);
    }
    return output;
  } else if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) && (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
    // We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
    // we track m1 vs m2 separately even though they must match for nicer error messages
    int64_t n = dim_tensor1 > 1 ? tensor1.size(-2) : 1;
    int64_t m1 = tensor1.size(-1);
    IntList batch_tensor1(tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0));
    int64_t m2 = dim_tensor2 > 1 ? tensor2.size(-2) : 1;
    int64_t p = tensor2.size(-1);
    IntList batch_tensor2(tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0));

    // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
    std::vector<int64_t> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);

    std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
    tensor1_expand_size.insert(tensor1_expand_size.end(), {n, m1});

    std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
    tensor2_expand_size.insert(tensor2_expand_size.end(), {m2, p});

    int expand_batch_product = std::accumulate(expand_batch_portion.begin(), expand_batch_portion.end(),
                                               1, std::multiplies<int64_t>());

    std::vector<int64_t> tensor1_bmm_view({expand_batch_product});
    tensor1_bmm_view.insert(tensor1_bmm_view.end(), {n, m1});

    std::vector<int64_t> tensor2_bmm_view({expand_batch_product});
    tensor2_bmm_view.insert(tensor2_bmm_view.end(), {m2, p});

    // flatten expanded batches
    Tensor tensor1_expanded = tensor1.expand(tensor1_expand_size).contiguous().view(tensor1_bmm_view);
    Tensor tensor2_expanded = tensor2.expand(tensor2_expand_size).contiguous().view(tensor2_bmm_view);

    Tensor output = tensor1_expanded.bmm(tensor2_expanded);

    // reshape batches back into result
    std::vector<int64_t> total_expansion(expand_batch_portion);
    total_expansion.insert(total_expansion.end(), {n, p});
    return maybeSqueeze(output.view(total_expansion), dim_tensor1, dim_tensor2);
  }

  runtime_error("both arguments to matmul need to be at least 1D, but they are %dD and %dD",
                dim_tensor1, dim_tensor2);

}

bool allclose(const Tensor& self, const Tensor& other, double rtol, double atol) {
  if (!self.sub(other).abs().le(other.abs().mul(rtol).add(atol)).all()) {
    return false;
  }

  return true;
}

template <typename scalar>
struct WhereOp {
  static void apply(Tensor& ret, const Tensor& condition, const Tensor& self, const Tensor& other) {
    CPU_tensor_apply4<scalar, uint8_t, scalar, scalar>(ret, condition, self, other,
      [](scalar& ret_val, const uint8_t& cond_val, const scalar& self_val, const scalar& other_val) {
        ret_val = cond_val ? self_val : other_val;
      }
    );
  }
};

Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  Tensor b_condition, b_self, b_other;
  std::tie(b_condition, b_self, b_other) = expand_outplace(condition, self, other, "where");
  return at::_s_where(b_condition, b_self, b_other);
}

Tensor _s_where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  if (condition.type().scalarType() != ScalarType::Byte) {
    runtime_error("Expected condition to have ScalarType Byte, but got ScalarType %s",
                  toString(condition.type().scalarType()));
  }
  if (self.type().backend() != Backend::CPU) {
    runtime_error("where() only supported with Backend::CPU, got %s", toString(self.type().backend()));
  }
  Tensor ret = self.type().tensor(self.sizes());
  dispatch_all<WhereOp>(ret.type(), "where", ret, condition, self, other);
  return ret;
}

std::tuple<at::Tensor, at::Tensor> RoiPooling2d_forward_cpu(
	const Tensor& input,
	const Tensor& rois,
	int64_t pooledHeight,
	int64_t pooledWidth,
	double spatialScale)
{
  // Input is the output of the last convolutional layer in the Backbone network, so
  // it should be in the format of NCHW
  AT_ASSERT(input.ndimension() == 4, "Input to RoI Pooling should be a NCHW Tensor");

  // ROIs is the set of region proposals to process. It is a 2D Tensor where the first
  // dim is the # of proposals, and the second dim is the proposal itself in the form
  // [batch_index startW startH endW endH]
  AT_ASSERT(rois.ndimension() == 2, "RoI Proposals should be a 2D Tensor, (batch_sz x proposals)");
  AT_ASSERT(rois.size(1) == 5, "Proposals should be of the form [batch_index startW startH endW enH]");

  auto proposals = rois.size(0);
  auto inputChannels = input.size(1);
  auto inputHeight = input.size(2);
  auto inputWidth = input.size(3);

  // Output Tensor is (num_rois, C, pooledHeight, pooledWidth)
  auto output = input.type().tensor({proposals, inputChannels, pooledHeight, pooledWidth});

  // TODO: need some mechanism for determining train vs. test

  // During training, we need to store the argmaxes for the pooling operation, so
  // the argmaxes Tensor should be the same size as the output Tensor
  auto argmaxes = input.type().toScalarType(kInt).tensor({proposals, inputChannels, pooledHeight, pooledWidth});

  AT_ASSERT(input.is_contiguous(), "input must be contiguous");
  AT_ASSERT(rois.is_contiguous(), "rois must be contiguous");

  auto *rawInput = input.data<float>();
  auto inputChannelStride = inputHeight * inputWidth;
  auto inputBatchStride = inputChannels * inputChannelStride;
  auto *rawRois = rois.data<float>();
  auto roiProposalStride = rois.size(1);

  auto *rawOutput = output.data<float>();
  auto *rawArgmaxes = argmaxes.data<int>();
  auto outputChannelStride = pooledHeight * pooledWidth;

  // Now that our Tensors are properly sized, we can perform the pooling operation.
  // We iterate over each RoI and perform pooling on each channel in the input, to
  // generate a pooledHeight x pooledWidth output for each RoI
  for (auto i = 0; i < proposals; ++i) {
    auto n = static_cast<int>(rawRois[0]);
    auto startWidth = static_cast<int>(std::round(rawRois[1] * spatialScale));
    auto startHeight = static_cast<int>(std::round(rawRois[2] * spatialScale));
    auto endWidth = static_cast<int>(std::round(rawRois[3] * spatialScale));
    auto endHeight = static_cast<int>(std::round(rawRois[4] * spatialScale));

    // TODO: assertions for valid values?
    // TODO: fix malformed ROIs??

    auto roiHeight = endHeight - startHeight;
    auto roiWidth = endWidth - startWidth;

    // Because the Region of Interest can be of variable size, but our output
    // must always be (pooledHeight x pooledWidth), we need to split the RoI
    // into a pooledHeight x pooledWidth grid of tiles

    auto tileHeight = static_cast<float>(roiHeight) / static_cast<float>(pooledHeight);
    auto tileWidth = static_cast<float>(roiWidth) / static_cast<float>(pooledWidth);

    auto *rawInputBatch = rawInput + (n * inputBatchStride);

    // Compute pooling for each of the (pooledHeight x pooledWidth) tiles for each
    // channel in the input
    for (auto ch = 0; ch < inputChannels; ++ch) {
      for (auto ph = 0; ph < pooledHeight; ++ph) {
        for (auto pw = 0; pw < pooledWidth; ++pw) {
          auto tileHStart = static_cast<int64_t>(std::floor(ph * tileHeight));
          auto tileWStart =	static_cast<int64_t>(std::floor(pw * tileWidth));
          auto tileHEnd = static_cast<int64_t>(std::ceil((ph + 1) * tileHeight));
          auto tileWEnd = static_cast<int64_t>(std::ceil((pw + 1) * tileWidth));

          // Add tile offsets to RoI offsets, and clip to input boundaries
          tileHStart = std::min(std::max<int64_t>(tileHStart + startHeight, 0), inputHeight);
          tileWStart = std::min(std::max<int64_t>(tileWStart + startWidth, 0), inputWidth);
          tileHEnd = std::min(std::max<int64_t>(tileHEnd + startHeight, 0), inputHeight);
          tileWEnd = std::min(std::max<int64_t>(tileWEnd + startWidth, 0), inputWidth);

          auto poolIndex = (ph * pooledWidth) + pw;

          // If our pooling region is empty, we set the output to 0, otherwise to
          // the min float so we can calculate the max properly
          auto empty = tileHStart >= tileHEnd || tileWStart >= tileWEnd;
          rawOutput[poolIndex] = empty ? 0 : std::numeric_limits<float>::min();

          // Set to -1 so we don't try to backprop to anywhere
          // TODO: make optional for test
          rawArgmaxes[poolIndex] = -1;

          for (auto th = tileHStart; th < tileHEnd; ++th) {
            for (auto tw = tileWStart; tw < tileWEnd; ++tw) {
              auto index = (th * inputWidth) + tw;
              if (rawInputBatch[index] > rawOutput[poolIndex]) {
                rawOutput[poolIndex] = rawInputBatch[index];
                // TODO: make optional for test
                rawArgmaxes[poolIndex] = index;
              }
            }
          }
        }
      }
      // Increment raw pointers by channel stride
      rawInputBatch += inputChannelStride;
      rawOutput += outputChannelStride;
      // TODO: make optional for test
      rawArgmaxes += outputChannelStride;
    }
    // Increment RoI raw pointer
    rawRois += roiProposalStride;
  }

  return std::make_tuple(output, argmaxes);
}

Tensor RoiPooling2d_backward_cpu(
  const Tensor& input,
  const Tensor& rois,
  int64_t pooledHeight,
  int64_t pooledWidth,
  double spatialScale,
  const Tensor& gradOutput,
  const Tensor& argmaxes) {
  throw std::runtime_error("not implemented");
}


// TODO Replace this with more accurate digamma().
template <typename scalar>
static inline scalar digamma_one(scalar x) {
  const scalar eps = x * 1e-2;
  return (std::lgamma(x + eps) - std::lgamma(x - eps)) / (eps + eps);
}

/** Computes the reparameterized gradient -(d/dalpha cdf(x;alpha)) / pdf(x;alpha)
    for random number x drawn from a standard Gamma distribution Gamma(alpha).
*/
template <typename scalar>
static inline scalar standard_gamma_grad_one(scalar alpha, scalar x) {
  // Use an asymptotic approximation for small x.
  if (x < 0.2f) {
    const auto a0 = 1 / alpha;
    const auto a1 = 1 / (alpha + 1);
    const auto a2 = 1 / (alpha + 2);
    const auto pow_x_alpha = std::pow(x, alpha);
    const auto gamma_pdf = std::pow(x, alpha - 1) * std::exp(-x);
    const auto gamma_cdf = pow_x_alpha * (a0 - x*a1 + 0.5f*x*x*a2);
    const auto gamma_cdf_alpha = (std::log(x) - digamma_one(alpha)) * gamma_cdf
        - pow_x_alpha * (a0*a0 - x*a1*a1 + 0.5f*x*x*a2*a2);
    const auto result = -gamma_cdf_alpha / gamma_pdf;
    return std::isnan(result) ? 0 : result;
  }

  // Use an asymptotic approximation for large alpha.
  if (alpha > 50.0f) {
    return std::sqrt(x / alpha);
  }

  // Use a bivariate rational approximation to the reparameterized gradient.
  const auto u = std::log(x / alpha);
  const auto v = std::log(alpha);
  static const scalar coef_uv[3][8] = {
    {0.16028008, -0.088064309, 0.019630876, -0.0016920282,
     1.0, 0.36659853, 0.10843863, 0.0066895454},
    {0.521894, 0.16095838, 0.06237597, 0.0023884253,
     0.083457714, 0.0073297628, -0.0059299053, -0.00093720389},
    {-0.0031143957, -0.012143877, -0.0057656484, -0.00064847254,
     0.0087262576, -0.00022820524, 1.8871047e-05, 9.6307964e-06},
  };
  scalar coef_v[8];
  for (int i = 0; i < 8; ++ i) {
    coef_v[i] = coef_uv[0][i] + u * (coef_uv[1][i] + u * coef_uv[2][i]);
  }
  const auto p = coef_v[0] + v * (coef_v[1] + v * (coef_v[2] + v * coef_v[3]));
  const auto q = coef_v[4] + v * (coef_v[5] + v * (coef_v[6] + v * coef_v[7]));
  return std::exp(p / q);
}

template <typename scalar>
struct StandardGammaGradOp {
  static void apply(Tensor& ret, const Tensor& self, const Tensor& output) {
    CPU_tensor_apply3<scalar, scalar, scalar>(ret, self, output,
      [](scalar& ret_val, const scalar& self_val, const scalar &output_val) {
         ret_val = standard_gamma_grad_one(self_val, output_val);
      }
    );
  }
};

Tensor _standard_gamma_grad(const Tensor& self, const Tensor& output) {
  Tensor ret = self.type().tensor(self.sizes());
  dispatch_floating_types<StandardGammaGradOp>(self.type(), "_standard_gamma_grad", ret, self, output);
  return ret;
}
  
Tensor conv_tbc(const Tensor& self, const Tensor& weight, const Tensor& bias, int64_t pad) {
  AT_ASSERT(self.dim() == 3, "Input must have 3 dims: time, batch, "
      "in_channel");
  AT_ASSERT(weight.dim() == 3, "Weight tensor must have 3 dims: kernel_width,"
      " in_channels, out_channels.");
  AT_ASSERT(bias.dim() == 1, "Bias must be 1-D");

  auto input_size = self.sizes();
  auto weight_size = weight.sizes();

  auto ilen = input_size[0];
  auto batchSize = input_size[1];
  auto inputPlanes = input_size[2];
  auto outputPlanes = weight_size[2];
  auto kw = weight_size[0];
  auto olen = input_size[0] - kw + 1 + pad * 2;
  auto real_pad = (olen - ilen + kw - 1) / 2;

  // Make sure shapes are correct.
  // Input = (time, batch, in_channels)
  // Weight = (kernel_width, in_channels, out_channels)
  // Bias = (out_channels)
  AT_ASSERT(inputPlanes == weight_size[1], "Input dim 2 (input channels) "
      "is not == dim 1 in the weight tensor");
  AT_ASSERT(weight_size[2] == bias.sizes()[0], "Bias size must equal dim 2 in "
      "the weight tensor (output channels).");

  // input * weights + bias -> output_features
  Tensor output = self.type().tensor({
    olen,
    input_size[1],
    weight_size[2],
  });
  output.copy_(bias.expand(output.sizes()));
  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, static_cast<int>(k - real_pad));
    int oShift = std::max(0, static_cast<int>(real_pad - k));
    int t = std::min(ilen + real_pad - k, olen) - oShift;
    // Note: gemm assumes column-major matrices
    // input    is l*m (row-major)
    // weight   is m*r (row-major)
    // output   is l*r (row-major)
    if (t > 0) {
      auto W = weight[k];
      auto I = self.narrow(0, iShift, t).view({t * batchSize, inputPlanes});
      auto O = output.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      O.addmm_(I, W);
    }
  }
  return output;
}

std::tuple<Tensor, Tensor, Tensor> conv_tbc_backward(const Tensor& dOutput, const Tensor& input, const Tensor& weight, const Tensor& bias, int64_t pad) {
  auto input_size = input.sizes();
  auto weight_size = weight.sizes();

  auto ilen = input_size[0];
  auto batchSize = input_size[1];
  auto inputPlanes = input_size[2];
  auto outputPlanes = weight_size[2];
  auto kw = weight.sizes()[0];
  auto olen = input_size[0] - kw + 1 + pad * 2;
  int real_pad = (olen - ilen + kw - 1) / 2;

  Tensor dInput = zeros_like(input);
  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - real_pad);
    int oShift = std::max(0, real_pad - k);
    int t = std::min(ilen + real_pad - k, olen) - oShift;
    // dOutput * T(weight) -> dInput
    if (t > 0) {
      auto dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      auto dI = dInput.narrow(0, iShift, t).view({t * batchSize, inputPlanes});
      dI.addmm_(dO, weight[k].t());
    }
  }

  Tensor dWeight = zeros_like(weight);
  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - real_pad);
    int oShift = std::max(0, real_pad - k);
    int t = std::min(ilen + real_pad - k, olen) - oShift;
    // T(input) * dOutput -> dWeight
    if (t > 0) {
      auto dW = dWeight[k];
      auto dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      auto I = input.narrow(0, iShift, t).view({t * batchSize, inputPlanes}).t();
      dW.addmm_(I, dO);
    }
  }

  Tensor dBias = zeros_like(bias); 
  auto tmp = dOutput.sum(0, false);
  dBias.assign_(tmp.sum(0));

  return std::make_tuple(dInput, dWeight, dBias);
}

}
}
