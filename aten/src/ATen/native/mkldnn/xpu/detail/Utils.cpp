#include <ATen/Context.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>

namespace at::native::onednn {

dnnl::memory make_onednn_memory(
    dnnl::memory::desc md,
    dnnl::engine& engine,
    void* ptr) {
  return dnnl::sycl_interop::make_memory(
      md,
      engine,
      dnnl::sycl_interop::memory_kind::usm,
      ptr == nullptr ? DNNL_MEMORY_ALLOCATE : ptr);
}

dnnl::memory::format_tag get_dnnl_default_format(
    int ndims,
    bool is_channels_last,
    bool allow_undef) {
  switch (ndims) {
    case 1:
      return dnnl::memory::format_tag::a;
    case 2:
      return dnnl::memory::format_tag::ab;
    case 3:
      return is_channels_last ? dnnl::memory::format_tag::acb
                              : dnnl::memory::format_tag::abc;
    case 4:
      return is_channels_last ? dnnl::memory::format_tag::acdb
                              : dnnl::memory::format_tag::abcd;
    case 5:
      return is_channels_last ? dnnl::memory::format_tag::acdeb
                              : dnnl::memory::format_tag::abcde;
    case 6:
      return dnnl::memory::format_tag::abcdef;
    case 7:
      return dnnl::memory::format_tag::abcdefg;
    case 8:
      return dnnl::memory::format_tag::abcdefgh;
    case 9:
      return dnnl::memory::format_tag::abcdefghi;
    case 10:
      return dnnl::memory::format_tag::abcdefghij;
    case 11:
      return dnnl::memory::format_tag::abcdefghijk;
    case 12:
      return dnnl::memory::format_tag::abcdefghijkl;
    default:
      if (!allow_undef) {
        TORCH_CHECK(false, "oneDNN doesn't support tensor dimension > 12");
      }
      return dnnl::memory::format_tag::undef;
  }
}

dnnl::memory::data_type get_onednn_dtype(
    const at::Tensor& tensor,
    bool allow_undef) {
  switch (tensor.scalar_type()) {
    case at::ScalarType::Byte:
      return dnnl::memory::data_type::u8;
    case at::ScalarType::Char:
      return dnnl::memory::data_type::s8;
    case at::ScalarType::QInt8:
      return dnnl::memory::data_type::s8;
    case at::ScalarType::QUInt8:
      return dnnl::memory::data_type::u8;
    case at::ScalarType::Int:
      return dnnl::memory::data_type::s32;
    case at::ScalarType::Half:
      return dnnl::memory::data_type::f16;
    case at::ScalarType::Float:
      return dnnl::memory::data_type::f32;
    case at::ScalarType::BFloat16:
      return dnnl::memory::data_type::bf16;
    default:
      if (!allow_undef) {
        TORCH_CHECK(
            false,
            c10::toString(tensor.scalar_type()),
            " is not supported in oneDNN!");
      }
      return dnnl::memory::data_type::undef;
  };
}

dnnl::memory::data_type get_onednn_dtype_include_double(
    const at::Tensor& tensor,
    bool allow_undef) {
  if (tensor.scalar_type() == at::ScalarType::Double)
    return dnnl::memory::data_type::f64;
  return get_onednn_dtype(tensor, allow_undef);
}

bool is_supported_onednn_dtype(const at::Tensor& tensor) {
  return get_onednn_dtype_include_double(tensor) !=
      dnnl::memory::data_type::undef;
}

dnnl::memory::dims get_onednn_dims(const at::Tensor& tensor) {
  dnnl::memory::dims dims;
  for (size_t i = 0; i < tensor.sizes().size(); i++)
    dims.push_back(tensor.size(i));
  return dims;
}

dnnl::memory::dims get_onednn_strides(const at::Tensor& tensor) {
  dnnl::memory::dims strides;
  for (size_t i = 0; i < tensor.strides().size(); i++)
    strides.push_back(tensor.stride(i));
  return strides;
}

dnnl::memory::desc get_onednn_md(const at::Tensor& tensor) {
  Tensor t = tensor.sizes().empty() ? tensor.unsqueeze(0) : tensor;
  return {
      get_onednn_dims(t),
      get_onednn_dtype_include_double(t),
      get_onednn_strides(t)};
}

bool onednn_strides_check(const Tensor& src) {
  auto adims = get_onednn_dims(src);
  int ndims = (int)adims.size();
  auto dims = adims.data();
  auto data_type = static_cast<dnnl_data_type_t>(
      get_onednn_dtype_include_double(src, /*allow_undef*/ false));
  auto strides_info = get_onednn_strides(src);
  auto strides = strides_info.empty() ? nullptr : &strides_info[0];

  dnnl_memory_desc_t md;
  dnnl_memory_desc_create_with_strides(&md, ndims, dims, data_type, strides);
  dnnl_format_kind_t md_fmt_kind;
  int md_ndims = 0;
  int md_inner_nblks = 0;
  dnnl_dims_t* md_padded_dims = nullptr;

  dnnl_memory_desc_query(md, dnnl_query_inner_nblks_s32, &md_inner_nblks);
  dnnl_memory_desc_query(md, dnnl_query_format_kind, &md_fmt_kind);
  dnnl_memory_desc_query(md, dnnl_query_ndims_s32, &md_ndims);
  dnnl_memory_desc_query(md, dnnl_query_padded_dims, &md_padded_dims);
  auto block_size = 1;
  // const auto& blk = md->format_desc.blocking;
  dnnl_dims_t md_inner_blks;
  dnnl_dims_t md_blk_inner_idxs;
  dnnl_memory_desc_query(md, dnnl_query_inner_idxs, &md_blk_inner_idxs);
  dnnl_memory_desc_query(md, dnnl_query_inner_blks, &md_inner_blks);
  dnnl_memory_desc_destroy(md);

  if (strides == nullptr || md_ndims == 0 ||
      md_fmt_kind != dnnl_format_kind_t::dnnl_blocked)
    return true;

  dnnl_dims_t blocks = {0};
  std::array<int, DNNL_MAX_NDIMS> perm = {0};
  for (int d = 0; d < md_ndims; ++d) {
    // no strides check needed for empty tensor
    if ((*md_padded_dims)[d] == 0)
      return true;

    // no strides verification for runtime dims
    if (strides[d] == DNNL_RUNTIME_DIM_VAL)
      return true;

    perm[d] = d;
    blocks[d] = 1;
  }

  for (int iblk = 0; iblk < md_inner_nblks; ++iblk) {
    blocks[md_blk_inner_idxs[iblk]] *= md_inner_blks[iblk];
    block_size *= md_inner_blks[iblk];
  }

  // A custom comparator to yield linear order on perm
  auto idx_sorter = [&](const int a, const int b) -> bool {
    if (strides[a] == strides[b] &&
        (*md_padded_dims)[a] == (*md_padded_dims)[b])
      return a < b;
    else if (strides[a] == strides[b])
      return (*md_padded_dims)[a] < (*md_padded_dims)[b];
    else
      return strides[a] < strides[b];
  };
  std::sort(perm.begin(), perm.begin() + md_ndims, idx_sorter);

  auto min_stride = block_size;
  for (int idx = 0; idx < md_ndims; ++idx) {
    const int d = perm[idx];

    // Make an exception for strides[d] == 0 as it has broadcast semantics
    // Note: owing to being sorted, these are the initial strides
    if (strides[d] == 0)
      continue;
    else if (strides[d] < min_stride)
      return false;

    // update min_stride for next iteration
    const auto padded_dim = (*md_padded_dims)[d];
    min_stride = block_size * strides[d] * (padded_dim / blocks[d]);
  }

  return true;
}

bool is_broadcast(const at::Tensor& t) {
  for (int i = 0; i < t.dim(); i++) {
    if (t.stride(i) == 0)
      return true;
  }
  return false;
}

void undo_broadcast_on_batch(at::Tensor& m1, at::Tensor& m2) {
  // oneDNN support one of src and wei broadcasted on batch dim
  // tensor shape = [b, m, n]
  constexpr int dim_b = 0;
  constexpr int dim_m = 1;
  constexpr int dim_n = 2;
  auto only_broadcasted_on_batch =
      [dim_b, dim_m, dim_n](const at::Tensor& tensor) {
        auto tensor_dim = tensor.dim();
        bool is_bmm = tensor_dim == 3;
        if (!is_bmm)
          return false;
        bool broadcast_on_mn =
            tensor.stride(dim_m) == 0 || tensor.stride(dim_n) == 0;
        bool has_broadcast_on_batch =
            tensor.stride(dim_b) == 0 && tensor.size(dim_b) > 1;
        // We do not support broadcast on dim m,n,k.
        // We can further optimize the case that both dim b and m are
        // broadcasted.
        if (broadcast_on_mn)
          return false;
        return has_broadcast_on_batch;
      };
  bool m1_only_batch_broadcasted = only_broadcasted_on_batch(m1);
  bool m2_only_batch_broadcasted = only_broadcasted_on_batch(m2);
  bool has_broadcast = m1_only_batch_broadcasted || m2_only_batch_broadcasted;
  bool both_broadcast = m1_only_batch_broadcasted && m2_only_batch_broadcasted;
  if (both_broadcast) {
    // oneDNN does not support both src and wei broadcasted on batch dim. We
    // copy the smaller one.
    if (m1.size(dim_m) < m2.size(dim_n)) {
      m1 = m1.contiguous();
      m1_only_batch_broadcasted = false;
    } else {
      m2 = m2.contiguous();
    }
  }
  if (has_broadcast) {
    at::Tensor& tensor = m1_only_batch_broadcasted ? m1 : m2;
    tensor = tensor
                 .as_strided(
                     {tensor.size(dim_m), tensor.size(dim_n)},
                     {tensor.stride(dim_m), tensor.stride(dim_n)})
                 .unsqueeze(dim_b);
  }
}

void undo_broadcast(at::Tensor& tensor) {
  // pytorch use stride = 0 for the dim to be broadcasted, but oneDNN only
  // support shape(dim) = 1 to implicitly indicate the broadcast dim.
  std::vector<int64_t> new_shape;
  std::vector<int64_t> new_strides;
  std::vector<int64_t> unsqueeze_dims;
  for (int i = 0; i < tensor.dim(); i++) {
    if (tensor.stride(i) == 0) {
      unsqueeze_dims.push_back(i);
    } else {
      new_shape.push_back(tensor.size(i));
      new_strides.push_back(tensor.stride(i));
    }
  }
  tensor = tensor.as_strided(new_shape, new_strides);
  for (size_t i = 0; i < unsqueeze_dims.size(); i++) {
    tensor = tensor.unsqueeze(unsqueeze_dims[i]);
  }
  return;
}

bool is_onednn_matmul_strides(const at::Tensor& tensor) {
  // https://oneapi-src.github.io/oneDNN/dev_guide_matmul.html
  // oneDNN matmul only support 2-dim and 3-dim
  // 2D src(Mxk), wei(KxN), dst(MxN)
  // 3D src(SxMxK), wei(WxKxN), dst(DxMxN)
  auto sizes = tensor.sizes();
  auto tensor_dim = sizes.size();
  if (tensor_dim != 2 && tensor_dim != 3)
    return false;

  if (tensor.is_contiguous())
    return true;

  if (tensor.storage_offset() > 0) {
    // currently onednn asks 64 byte alignment
    constexpr int alignment_byte = 64;
    if (reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment_byte > 0)
      return false;
  }

  // the overlapped cases are not supported
  dnnl::memory::dims strides = get_onednn_strides(tensor);
  int64_t storage_size = 1;
  for (size_t dim = 0; dim < tensor_dim; ++dim)
    storage_size += (sizes[dim] - 1) * strides[dim];
  if (storage_size < tensor.numel())
    return false;

  // the broadcast cases are not supported
  if (is_broadcast(tensor)) {
    return false;
  }
  // the src and weight must have at least one of the axes
  // m or k and n or k contiguous (i.e., stride=1) respectively.
  if (strides[tensor_dim - 1] != 1 && strides[tensor_dim - 2] != 1)
    return false;

  if (!onednn_strides_check(tensor))
    return false;
  return true;
}

bool is_broadcast_from_other_to_self(
    const at::Tensor& self,
    const at::Tensor& other) {
  return (
      self.sizes() != other.sizes() &&
      at::is_expandable_to(other.sizes(), self.sizes()));
}

at::MemoryFormat get_cl_tag_by_ndim(const int64_t ndim) {
  TORCH_CHECK(
      3 == ndim || 4 == ndim || 5 == ndim,
      "ndim must be 3, 4 or 5 when get cl tag");
  if (3 == ndim) {
    return at::MemoryFormat::Contiguous;
  } else if (5 == ndim) {
    return at::MemoryFormat::ChannelsLast3d;
  } else {
    return at::MemoryFormat::ChannelsLast;
  }
}

bool binary_valid(
    const at::Tensor& self,
    const at::Tensor& other,
    bool is_fusion) {
  if (self.sizes() != other.sizes() &&
      !is_broadcast_from_other_to_self(self, other))
    return false;

  /* If the following conditions are satisfied, then oneDNN path will be
     selected:
     * 1. self and other should be xpu tensor and be defined.
     * 2. self or other should not be scalar (wrapped tensor).
     * 3. dim of self and other should be equal and must be larger than 0 and
     smaller than 7.
     * 4. the datatype should be supported by oneDNN primitive.
     * 5. self and other should be in the same datatype.
     * 6. self and other should be contiguous or channel-last contiguous.*/

  // 1. self and other should be xpu tensor and be defined.
  if ((!self.defined()) || (!other.defined()) || (!self.is_xpu()) ||
      (!other.is_xpu()))
    return false;

  // 2. self or other should not be scalar (wrapped tensor).
  if (self.unsafeGetTensorImpl()->is_wrapped_number() ||
      other.unsafeGetTensorImpl()->is_wrapped_number())
    return false;

  // 3. dim of self and other should be equal and must be larger than 0 and
  // smaller than 7.
  if ((self.dim() <= 0) || (other.dim() <= 0) || (self.dim() != other.dim()) ||
      (self.dim() > 6) || (other.dim() > 6))
    return false;

  // 4. the datatype should be supported by oneDNN primitive.
  switch (self.scalar_type()) {
    case at::ScalarType::Char:
      break;
    case at::ScalarType::Byte:
      break;
    case at::ScalarType::Half:
      break;
    case at::ScalarType::Float:
      break;
    case at::ScalarType::BFloat16:
      break;
    default:
      return false;
  };

  // 5. datatype check
  if (is_fusion) {
    // for fusion case, the fusion can be performed on scalar_type or Float
    // datatype.
    if (self.scalar_type() != other.scalar_type() &&
        other.scalar_type() != at::ScalarType::Float) {
      return false;
    }
  } else {
    if (self.scalar_type() != other.scalar_type()) {
      // for non-fusion case: self and other should be in the same datatype.
      return false;
    }
  }

  // 6. self and other should be contiguous or channel-last contiguous.
  const auto ndim = self.ndimension();
  auto cl_tag = at::MemoryFormat::ChannelsLast;
  if (3 == ndim || 4 == ndim || 5 == ndim) {
    cl_tag = get_cl_tag_by_ndim(ndim);
  }
  if ((self.is_contiguous() && other.is_contiguous()) ||
      (self.is_contiguous(cl_tag) && other.is_contiguous(cl_tag)))
    return true;
  return false;
}

static inline bool is_channels_last(at::MemoryFormat fmt) {
  return (at::MemoryFormat::ChannelsLast == fmt) ||
      (at::MemoryFormat::ChannelsLast3d == fmt);
}

static inline bool is_smf_channels_last(const Tensor& t) {
  return is_channels_last(t.suggest_memory_format());
}

bool use_channels_last_for_conv(
    const at::Tensor& src,
    const at::Tensor& weight) {
  return xpu_conv_use_channels_last(src, weight);
}

dnnl::memory::format_tag conv_src_fmt(
    const int64_t ndim,
    const bool is_channels_last) {
  if (!is_channels_last) {
    return (ndim == 3)
        ? dnnl::memory::format_tag::ncw
        : ((ndim == 4) ? dnnl::memory::format_tag::nchw
                       : ((ndim == 5) ? dnnl::memory::format_tag::ncdhw
                                      : dnnl::memory::format_tag::undef));
  } else {
    return (ndim == 3)
        ? dnnl::memory::format_tag::nwc
        : ((ndim == 4) ? dnnl::memory::format_tag::nhwc
                       : ((ndim == 5) ? dnnl::memory::format_tag::ndhwc
                                      : dnnl::memory::format_tag::undef));
  }
}

dnnl::memory::dims compatible_weight_dims(
    const int64_t ndim,
    const int64_t groups,
    const int64_t oc,
    const int64_t ic,
    const IntArrayRef wsizes) {
  if (ndim == 3) {
    auto kw = wsizes[2];
    return (groups != 1)
        ? dnnl::memory::dims({groups, oc / groups, ic / groups, kw})
        : dnnl::memory::dims({oc, ic, kw});
  } else if (ndim == 4) {
    auto kh = wsizes[2];
    auto kw = wsizes[3];
    return (groups != 1)
        ? dnnl::memory::dims({groups, oc / groups, ic / groups, kh, kw})
        : dnnl::memory::dims({oc, ic, kh, kw});
  } else if (ndim == 5) {
    auto kd = wsizes[2];
    auto kh = wsizes[3];
    auto kw = wsizes[4];
    return (groups != 1)
        ? dnnl::memory::dims({groups, oc / groups, ic / groups, kd, kh, kw})
        : dnnl::memory::dims({oc, ic, kd, kh, kw});
  }

  return {};
}

dnnl::memory::format_tag conv_weight_fmt(
    const int64_t ndim,
    const bool grouped,
    const bool is_channels_last) {
  if (!is_channels_last) {
    return (ndim == 3) ? (grouped ? dnnl::memory::format_tag::goiw
                                  : dnnl::memory::format_tag::oiw)
        : (ndim == 4)
        ? (grouped ? dnnl::memory::format_tag::goihw
                   : dnnl::memory::format_tag::oihw)
        : ((ndim == 5) ? (grouped ? dnnl::memory::format_tag::goidhw
                                  : dnnl::memory::format_tag::oidhw)
                       : dnnl::memory::format_tag::undef);
  } else {
    return (ndim == 3) ? (grouped ? dnnl::memory::format_tag::gowi
                                  : dnnl::memory::format_tag::owi)
        : (ndim == 4)
        ? (grouped ? dnnl::memory::format_tag::gohwi
                   : dnnl::memory::format_tag::ohwi)
        : ((ndim == 5) ? (grouped ? dnnl::memory::format_tag::godhwi
                                  : dnnl::memory::format_tag::odhwi)
                       : dnnl::memory::format_tag::undef);
  }
}

void apply_tf32_if_allowed(dnnl::primitive_attr& pattr) {
  auto& ctx = at::globalContext();
  bool allow_tf32 = ctx.allowTF32OneDNN();
  if (allow_tf32) {
    pattr.set_fpmath_mode(dnnl::fpmath_mode::tf32);
  }
}

} // namespace at::native::onednn
