#pragma once


#include <iostream>


#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/core/Tensor.h>

#include <ATen/core/grad_mode.h>
// #include <aten/quantized/QUtils.h>
// #include <aten/quantized/Quantizer.h>
#include <c10/core/MemoryFormat.h>
// #include <core/detail/TensorInfo.h>
// #include <oneDNN/Runtime.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
// #include <runtime/Utils.h>
// #include <tensor/Context.h>
// #include <utils/Macros.h>
// #include <utils/Settings.h>

using namespace dnnl;

#define DPCPP_ONEDNN_EXEC(prim, stream, ...)                           \
  {                                                                    \
    auto q = dnnl::sycl_interop::get_queue((stream));                  \
    DPCPP_EXT_SUBMIT(                                                  \
        (q),                                                           \
        "onednn_kernel",                                               \
        dnnl::sycl_interop::execute((prim), (stream), ##__VA_ARGS__)); \
  }

// FIXME: In some cases, for example, concat, reorder, and etc.
// oneDNN only supports dims <= 6 for now.
#define MAX_ONEDNN_SUPPORTED_DIMS 6

// sc&zp always have same md for PerTensor Quantization
// Following two mds would have external linkage, but with same
// address in different compiation unit.
inline memory::desc Q_PER_TENSOR_SC_MD =
    memory::desc({1}, memory::data_type::f32, memory::format_tag::x);
inline memory::desc Q_PER_TENSOR_ZP_MD =
    memory::desc({1}, memory::data_type::s32, memory::format_tag::x);

namespace xpu {
namespace oneDNN {

// static inline std::pair<memory, memory> q_get_sc_zp_gpu_mem(
//     const Tensor& qx,
//     dnnl::engine& engine) {
//   memory qx_sc_m, qx_zp_m;
//   using xpu::dpcpp::XPUQuantizerBase;
//   xpu::dpcpp::lru_key_t key_sc_zp;
//   auto quant_base =
//       xpu::dpcpp::fetch_cached_quantizer_base(qx.q_scale(), qx.q_zero_point());
//   auto sc_ptr = quant_base.scale_ptr();
//   auto zp_ptr = quant_base.zero_point_ptr();
//   qx_sc_m = dpcpp_onednn_memory(Q_PER_TENSOR_SC_MD, engine, sc_ptr);
//   qx_zp_m = dpcpp_onednn_memory(Q_PER_TENSOR_ZP_MD, engine, zp_ptr);
//   return {qx_sc_m, qx_zp_m};
// }

// static inline memory q_get_wgh_sc_gpu_mem(
//     const Tensor& qw,
//     dnnl::engine& engine) {
//   float dnn_scale = qw.q_scale();
//   auto quant_base = xpu::dpcpp::fetch_cached_quantizer_base(dnn_scale, 0);
//   auto sc_ptr = quant_base.scale_ptr();
//   memory sc_m = dpcpp_onednn_memory(Q_PER_TENSOR_SC_MD, engine, sc_ptr);
//   return sc_m;
// }

static inline memory::format_tag get_dnnl_default_format(
    int ndims,
    bool is_channels_last = false,
    bool allow_undef = false) {
  switch (ndims) {
    case 1:
      return memory::format_tag::a;
    case 2:
      return memory::format_tag::ab;
    case 3:
      return is_channels_last ? memory::format_tag::acb
                              : memory::format_tag::abc;
    case 4:
      return is_channels_last ? memory::format_tag::acdb
                              : memory::format_tag::abcd;
    case 5:
      return is_channels_last ? memory::format_tag::acdeb
                              : memory::format_tag::abcde;
    case 6:
      return memory::format_tag::abcdef;
    case 7:
      return memory::format_tag::abcdefg;
    case 8:
      return memory::format_tag::abcdefgh;
    case 9:
      return memory::format_tag::abcdefghi;
    case 10:
      return memory::format_tag::abcdefghij;
    case 11:
      return memory::format_tag::abcdefghijk;
    case 12:
      return memory::format_tag::abcdefghijkl;
    default:
      if (!allow_undef) {
        TORCH_CHECK(false, "oneDNN doesn't support tensor dimension > 12");
      }
      return memory::format_tag::undef;
  }
}

static inline memory::data_type get_onednn_dtype(
    const at::Tensor& tensor,
    bool allow_undef = false) {
  switch (tensor.scalar_type()) {
    case at::ScalarType::Byte:
      return memory::data_type::u8;
    case at::ScalarType::Char:
      return memory::data_type::s8;
    case at::ScalarType::QInt8:
      return memory::data_type::s8;
    case at::ScalarType::QUInt8:
      return memory::data_type::u8;
    case at::ScalarType::Int:
      return memory::data_type::s32;
    case at::ScalarType::Half:
      return memory::data_type::f16;
    case at::ScalarType::Float:
      return memory::data_type::f32;
    case at::ScalarType::BFloat16:
      return memory::data_type::bf16;
    default:
      if (!allow_undef) {
        TORCH_CHECK(
            false,
            c10::toString(tensor.scalar_type()),
            " is not supported in oneDNN!");
      }
      return memory::data_type::undef;
  };
}

static inline memory::data_type get_onednn_dtype_include_double(
    const at::Tensor& tensor,
    bool allow_undef = false) {
  if (tensor.scalar_type() == at::ScalarType::Double)
    return memory::data_type::f64;
  return get_onednn_dtype(tensor, allow_undef);
}

static bool is_supported_onednn_dtype(const at::Tensor& tensor) {
  return get_onednn_dtype(tensor, /*allow_undef*/ true) ==
          memory::data_type::undef
      ? false
      : true;
}

// Here this function is used to deduce the torch tensor meta dtype from the
// kept oqaque tensor context in case of saving tensor
// static inline c10::ScalarType opaqueTypeToScalarType(const at::Tensor& tensor) {
//   auto is_quantized = tensor.is_quantized();
//   auto ctx = *(static_cast<at::AtenIpexTypeXPU::DPCPPTensorContext*>(
//       tensor.unsafeGetTensorImpl()->storage().data_ptr().get_context()));
//   switch (ctx.dtype()) {
//     case dnnl::memory::data_type::u8:
//       // For quantized tensor, the meta dtype is QUInt8
//       return (is_quantized) ? at::ScalarType::QUInt8 : at::ScalarType::Byte;
//     case dnnl::memory::data_type::s8:
//       // For quantized tensor, the meta dtype is QInt8
//       return (is_quantized) ? at::ScalarType::QInt8 : at::ScalarType::Char;
//     case dnnl::memory::data_type::f16:
//       return at::ScalarType::Half;
//     case dnnl::memory::data_type::f32:
//       return at::ScalarType::Float;
//     case dnnl::memory::data_type::bf16:
//       return at::ScalarType::BFloat16;
//     case dnnl::memory::data_type::f64:
//       return at::ScalarType::Double;
//     default:
//       TORCH_CHECK(false, "Cannot be translated to torch dtype");
//   };
// }

// static inline bool check_equality_for_meta_dtype_and_ctx_dtype(
//     const at::Tensor& tensor) {
//   auto ctx_dtype = opaqueTypeToScalarType(tensor);
//   return bool(ctx_dtype == tensor.scalar_type());
// }

// static inline fpmath_mode get_onednn_fpmath_mode() {
//   auto math_mode = Settings::I().get_fp32_math_mode();
//   switch (math_mode) {
//     case FP32_MATH_MODE::TF32:
//       return fpmath_mode::tf32;
//     case FP32_MATH_MODE::BF32:
//       return fpmath_mode::bf16;
//     default: // use FP32_MATH_MODE::FP32 as default
//       return fpmath_mode::strict;
//   }
// }

static inline memory::dims get_onednn_dims(const at::Tensor& tensor) {
  memory::dims dims;
  for (int i = 0; i < tensor.sizes().size(); i++)
    dims.push_back(tensor.size(i));
  return dims;
}

static inline memory::dims get_onednn_strides(const at::Tensor& tensor) {
  memory::dims strides;
  for (int i = 0; i < tensor.strides().size(); i++)
    strides.push_back(tensor.stride(i));
  return strides;
}

static inline memory::desc get_onednn_md(const at::Tensor& tensor) {
  return {
      get_onednn_dims(tensor),
      get_onednn_dtype(tensor),
      get_onednn_strides(tensor)};
}

template <typename T>
inline void array_copy(T* dst, const T* src, size_t size) {
  for (size_t i = 0; i < size; ++i)
    dst[i] = src[i];
}

inline bool onednn_strides_check(const at::Tensor& src) {
  auto adims = xpu::oneDNN::get_onednn_dims(src);
  int ndims = (int)adims.size();
  auto dims = adims.data();
  auto data_type = static_cast<dnnl_data_type_t>(
      xpu::oneDNN::get_onednn_dtype(src, /*allow_undef*/ true));
  auto strides_info = xpu::oneDNN::get_onednn_strides(src);
  auto strides = strides_info.empty() ? nullptr : &strides_info[0];

  dnnl_memory_desc_t md;
  dnnl_memory_desc_create_with_strides(&md, ndims, dims, data_type, strides);
  // md->get_ndims() = ndims;
  // array_copy(md->dims, dims, ndims);
  // md->get_data_type() = data_type;
  // array_copy(md->padded_dims, dims, ndims);
  // md->get_format_kind() = dnnl_format_kind_t::dnnl_blocked;
  dnnl_format_kind_t md_fmt_kind;
  int md_ndims;
  int md_inner_nblks;
  dnnl_dims_t* md_padded_dims = nullptr;

  dnnl_memory_desc_query(md, dnnl_query_inner_nblks_s32, &md_inner_nblks);
  dnnl_memory_desc_query(md, dnnl_query_format_kind, &md_fmt_kind);
  dnnl_memory_desc_query(md, dnnl_query_ndims_s32, &md_ndims);
  dnnl_memory_desc_query(md, dnnl_query_padded_dims, &md_padded_dims);
  if (strides == nullptr || md_ndims == 0 ||
      md_fmt_kind != dnnl_format_kind_t::dnnl_blocked)
    return true;

  dnnl_dims_t blocks = {0};
  int perm[DNNL_MAX_NDIMS] = {0};
  for (int d = 0; d < md_ndims; ++d) {
    // no strides check needed for empty tensor
    if (md_padded_dims[d] == 0)
      return true;

    // no strides verification for runtime dims
    if (strides[d] == DNNL_RUNTIME_DIM_VAL)
      return true;

    perm[d] = d;
    blocks[d] = 1;
  }

  auto block_size = 1;
  // const auto& blk = md->format_desc.blocking;
  dnnl_dims_t md_inner_blks;
  dnnl_dims_t md_blk_inner_idxs;
  dnnl_memory_desc_query(md, dnnl_query_inner_idxs, &md_blk_inner_idxs);
  dnnl_memory_desc_query(md, dnnl_query_inner_blks, &md_inner_blks);
  for (int iblk = 0; iblk < md_inner_nblks; ++iblk) {
    blocks[md_blk_inner_idxs[iblk]] *= md_inner_blks[iblk];
    block_size *= md_inner_blks[iblk];
  }

  // A custom comparator to yield linear order on perm
  auto idx_sorter = [&](const int a, const int b) -> bool {
    if (strides[a] == strides[b] && md_padded_dims[a] == md_padded_dims[b])
      return a < b;
    else if (strides[a] == strides[b])
      return md_padded_dims[a] < md_padded_dims[b];
    else
      return strides[a] < strides[b];
  };
  std::sort(perm, perm + md_ndims, idx_sorter);

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
    const auto padded_dim = *md_padded_dims[d];
    min_stride = block_size * strides[d] * (padded_dim / blocks[d]);
  }
  return true;
}

static inline bool is_broadcast(const at::Tensor& t) {
  for (int i = 0; i < t.dim(); i++) {
    if (t.stride(i) == 0)
      return true;
  }
  return false;
}

static inline bool is_onednn_matmul_strides(
    const at::Tensor& tensor,
    bool is_dst = false) {
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

  // the overlaped cases are not supported
  memory::dims strides = get_onednn_strides(tensor);
  int64_t storage_size = 1;
  for (size_t dim = 0; dim < tensor_dim; ++dim)
    storage_size += (sizes[dim] - 1) * strides[dim];
  if (storage_size < tensor.numel())
    return false;

  // the broadcast cases are not supported
  if (is_broadcast(tensor)) {
    return false;
  }

  if (is_dst) {
    // The memory format of the destination tensor should always
    // be plain with n axis contiguous
    if (strides[-1] != 1)
      return false;
  } else {
    // the src and weight must have at least one of the axes
    // m or k and n or k contiguous (i.e., stride=1) respectively.
    if (strides[tensor_dim - 1] != 1 && strides[tensor_dim - 2] != 1)
      return false;
  }

  if (!onednn_strides_check(tensor))
    return false;

  return true;
}

// static inline std::vector<int64_t> compatible_groups_conv_strides(
//     const at::Tensor& wgh,
//     memory::dims group_size) {
//   std::vector<int64_t> strides = wgh.strides().vec();
//   strides.insert(strides.begin(), group_size[1] * wgh.stride(0));
//   return strides;
// }
// static inline std::vector<int64_t> compatible_groups_deconv_strides(
//     const at::Tensor& wgh,
//     memory::dims group_size) {
//   std::vector<int64_t> strides = wgh.strides().vec();
//   strides[0] = wgh.strides()[1];
//   strides[1] = wgh.strides()[0];
//   strides.insert(strides.begin(), group_size[2] * wgh.strides()[0]);
//   return strides;
// }

// static inline bool is_onednn_layout(const at::Tensor& tensor) {
//   return !at::AtenIpexTypeXPU::DPCPPTensorContext::is_plain(tensor);
// }

// template <typename T>
// static inline bool iteratable_has_onednn_layout(T container) {
//   bool has_onednn_layout_tensor =
//       std::any_of(container.begin(), container.end(), [](const Tensor& t) {
//         return xpu::oneDNN::is_onednn_layout(t);
//       });
//   return has_onednn_layout_tensor;
// }

// static inline bool has_onednn_layout(const Tensor& tensor) {
//   return is_onednn_layout(tensor);
// }

// static inline bool has_onednn_layout(const TensorList& inputs) {
//   return iteratable_has_onednn_layout(inputs);
// }

// static inline bool has_onednn_layout(const ITensorListRef& inputs) {
//   auto tensors = inputs.materialize();
//   return iteratable_has_onednn_layout(tensors);
// }

// T would be Tensor, TensorList, ITensorListRef
// template <typename T, typename... Args>
// bool has_onednn_layout(const T& input, const Args&... rest) {
//   if (has_onednn_layout(input)) {
//     return true;
//   }
//   return false || has_onednn_layout(rest...);
// }

// static inline bool eltwise_forward_valid(const at::Tensor& tensor) {
//   switch (tensor.scalar_type()) {
//     // return false if scalar_type not supported
//     case at::ScalarType::Float:
//       break;
//     case at::ScalarType::BFloat16:
//       break;
//     case at::ScalarType::Half:
//       break;
//     case at::ScalarType::Int:
//       break;
//     case at::ScalarType::Char:
//       break;
//     case at::ScalarType::Byte:
//       break;
//     default:
//       return false;
//   };
//   if (tensor.dim() > 6)
//     return false;
//   if (!at::AtenIpexTypeXPU::DPCPPTensorContext::is_plain(tensor))
//     return true;
//   if (tensor.is_contiguous() || tensor.dim() == 1)
//     return true;
//   return false;
// }

// static inline bool eltwise_backward_valid(const at::Tensor& tensor) {
//   switch (tensor.scalar_type()) {
//     case at::ScalarType::Float:
//       break;
//     case at::ScalarType::BFloat16:
//       break;
//     default:
//       return false;
//   };
//   if (tensor.dim() > 6)
//     return false;
//   if (!at::AtenIpexTypeXPU::DPCPPTensorContext::is_plain(tensor))
//     return true;
//   if (tensor.is_contiguous() || tensor.dim() == 1)
//     return true;
//   return false;
// }

static bool is_wrapped_number(const at::Tensor& t) {
  return t.unsafeGetTensorImpl()->is_wrapped_number();
}

static inline bool is_broadcast_from_other_to_self(
    const at::Tensor& self,
    const at::Tensor& other) {
  return (
      self.sizes() != other.sizes() &&
      at::is_expandable_to(other.sizes(), self.sizes()));
}


inline at::MemoryFormat get_cl_tag_by_ndim(const int64_t ndim) {
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

static inline bool binary_valid(
    const at::Tensor& self,
    const at::Tensor& other,
    bool is_fusion = false) {
  // FIXME: update onednn
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
  if (is_wrapped_number(self) || is_wrapped_number(other))
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

// static inline bool softmax_valid(const at::Tensor& self) {
//   if (!self.is_contiguous())
//     return false;

//   if (self.sizes().size() > 4 || self.sizes().size() < 1)
//     return false;

//   // the datatype should be supported by oneDNN primitive.
//   switch (self.scalar_type()) {
//     case at::ScalarType::Half:
//       break;
//     case at::ScalarType::Float:
//       break;
//     case at::ScalarType::BFloat16:
//       break;
//     default:
//       return false;
//   };
//   return true;
// }

// static inline bool softmax_backward_valid(
//     const at::Tensor& grad,
//     const at::Tensor& output,
//     const at::Tensor& input) {
//   if (!grad.is_contiguous() || !output.is_contiguous())
//     return false;

//   if (input.sizes().size() > 4 || input.sizes().size() < 1)
//     return false;

//   // the datatype should be supported by oneDNN primitive.
//   switch (input.scalar_type()) {
//     case at::ScalarType::Float:
//       break;
//     case at::ScalarType::BFloat16:
//       break;
//     default:
//       return false;
//   };
//   return true;
// }

// static inline bool cat_valid(const TensorList& tensors) {
//   for (int i = 0; i < tensors.size(); i++) {
//     const Tensor& tensor = tensors[i];
//     if (tensor.defined()) {
//       if (tensor.scalar_type() == ScalarType::Bool ||
//           tensor.scalar_type() == ScalarType::Short ||
//           tensor.scalar_type() == ScalarType::Double ||
//           tensor.scalar_type() == ScalarType::Long ||
//           tensor.scalar_type() == ScalarType::ComplexFloat ||
//           tensor.scalar_type() == ScalarType::ComplexDouble ||
//           tensor.dim() > MAX_ONEDNN_SUPPORTED_DIMS) {
//         return false;
//       }
//     }
//   }
//   return true;
// }

enum MEMORY_LAYOUT_FOR_CONV {
  ChannelsFirst = 0, // using channels_first for conv computation.
  ChannelsLast = 1, /// using channels_last for conv computation.
  Blocked = 2, // using blocked format for conv computation.
};

// static inline int get_memory_layout_for_conv(
//     const at::Tensor& src,
//     const at::Tensor& weight,
//     bool is_transpose) {
//   if (!src.defined() || src.is_sparse()) {
//     // suggest channels_first
//     return MEMORY_LAYOUT_FOR_CONV::ChannelsFirst;
//   }

//   if (is_transpose || src.is_quantized() || weight.is_quantized() ||
//       (!dpcppSupportFP64())) {
//     if (Settings::I().is_onednn_layout_enabled()) {
//       // suggest blocked
//       return MEMORY_LAYOUT_FOR_CONV::Blocked;
//     }
//   }

//   auto suggest_channels_last_format =
//       (is_smf_channels_last(src) || is_smf_channels_last(weight));
//   if (suggest_channels_last_format) {
//     // suggest channels_last
//     return MEMORY_LAYOUT_FOR_CONV::ChannelsLast;
//   }

//   // inference workloads on ATSM platform, the conv will use blocked format
//   // used double support to distinguish is atsm or not
//   auto suggest_block_format = !dpcppSupportFP64() // on ATSM platform
//       && (c10::InferenceMode::is_enabled() ||
//           !at::GradMode::is_enabled()); // for inference workload
//   if (suggest_block_format) {
//     // suggest blocked
//     return MEMORY_LAYOUT_FOR_CONV::Blocked;
//   }

//   // suggest channels_last
//   return MEMORY_LAYOUT_FOR_CONV::ChannelsFirst;
// }

// static inline at::MemoryFormat get_tensor_format_for_conv(
//     const at::Tensor& src,
//     const at::Tensor& weight,
//     bool is_transposed) {
//   at::MemoryFormat mfmt;
//   if (get_memory_layout_for_conv(src, weight, is_transposed) ==
//       MEMORY_LAYOUT_FOR_CONV::ChannelsLast) {
//     mfmt = get_cl_tag_by_ndim(src.ndimension());
//   } else {
//     mfmt = at::MemoryFormat::Contiguous;
//   }
//   return mfmt;
// }

// judge to use block or plain for Matmul
static inline bool using_onednn_layout_for_matmul(const at::Tensor& src) {
  // if (!src.defined() || src.is_sparse()) {
  //   // suggest plain
  //   return false;
  // }

  // if (Settings::I().is_onednn_layout_enabled()) {
  //   // suggest block
  //   return true;
  // }

  // auto src_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src);
  // if (!src_ctx.is_plain()) {
  //   // suggest block
  //   return true;
  // }

  // suggest plain
  return false;
}

// static inline bool using_channels_last_for_onednn_op(const at::Tensor& input) {
//   const auto ndim = input.ndimension();
//   if (ndim == 2) {
//     return false;
//   }

//   // if input is blocked format, then pooling will use blocked instead of plain
//   // format
//   auto input_ctx =
//       at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(input);
//   if (!input_ctx.is_plain()) {
//     return false;
//   }

//   return is_smf_channels_last(input);
// }

static inline at::Tensor contiguous_if_needed(
    const at::Tensor& t,
    at::MemoryFormat mfmt = at::MemoryFormat::Contiguous) {
  // auto ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(t);
  // Tensor t_ = ctx.is_plain() ? t.contiguous(mfmt) : t;
  at::Tensor t_ = t.contiguous(mfmt);
  return t_;
}

// static inline bool eltwise_forward_valid(
//     const Tensor& out,
//     const Tensor& self) {
//   bool onednn_path_valid = true;
//   if (!(is_onednn_layout(self) && eltwise_forward_valid(self))) {
//     onednn_path_valid = false;
//   }
//   if (!out.defined()) {
//     return onednn_path_valid;
//   } else {
//     if (!out.is_view() && out.is_contiguous() &&
//         self.scalar_type() == out.scalar_type()) {
//       // The output tensor is not a slice of another tensor
//       return onednn_path_valid;
//     } else {
//       // The output tensor is a slice of another tensor
//       TORCH_CHECK(
//           !xpu::oneDNN::is_onednn_layout(out),
//           "cannot convert tensor slice to plain format");
//       return false;
//     }
//   }
// }

// static inline bool eltwise_backward_valid(
//     const Tensor& out,
//     const Tensor& self,
//     const Tensor& other) {
//   bool onednn_path_valid = true;
//   if (!(is_onednn_layout(self) && is_onednn_layout(other) &&
//         eltwise_backward_valid(self) && eltwise_backward_valid(other))) {
//     onednn_path_valid = false;
//   }
//   if (!out.defined()) {
//     return onednn_path_valid;
//   } else {
//     if (!out.is_view() && out.is_contiguous() &&
//         self.scalar_type() == out.scalar_type()) {
//       // The output tensor is not a slice of another tensor
//       return onednn_path_valid;
//     } else {
//       // The output tensor is a slice of another tensor
//       TORCH_CHECK(
//           !xpu::oneDNN::is_onednn_layout(out),
//           "cannot convert tensor slice to plain format");
//       return false;
//     }
//   }
// }

// static inline bool binary_forward_valid(
//     const Tensor& out,
//     const Tensor& self,
//     const Tensor& other) {
//   bool onednn_path_valid = true;
//   if (!(IPEX_ANY(xpu::oneDNN::is_onednn_layout, self, other) &&
//         binary_valid(self, other))) {
//     onednn_path_valid = false;
//   }
//   if (!out.defined()) {
//     return onednn_path_valid;
//   } else {
//     if (!out.is_view() && out.is_contiguous() &&
//         self.scalar_type() == out.scalar_type()) {
//       // The output tensor is not a slice of another tensor
//       return onednn_path_valid;
//     } else {
//       // The output tensor is a slice of another tensor
//       TORCH_CHECK(
//           !xpu::oneDNN::is_onednn_layout(out),
//           "cannot convert tensor slice to plain format");
//       return false;
//     }
//   }
// }

// template <typename T>
// static void print_vec_xpu(const char* str, T* vec, int size) {
//   printf("%s", str);

//   for (int d = 0; d < size; ++d)
//     printf("%d ", (int)vec[d]);

//   printf("\n");
// }

// static void dump_md_data_type(memory::data_type dt) {
//   switch (dt) {
//     case memory::data_type::undef:
//       std::cout << "data_type undef";
//       break;
//     case memory::data_type::f16:
//       std::cout << "data type:f16" << std::endl;
//       break;
//     case memory::data_type::bf16:
//       std::cout << "data type:bf16" << std::endl;
//       break;
//     case memory::data_type::f32:
//       std::cout << "data type:f32" << std::endl;
//       break;
//     case memory::data_type::f64:
//       std::cout << "data type:f64" << std::endl;
//       break;
//     case memory::data_type::s32:
//       std::cout << "data type:s32" << std::endl;
//       break;
//     case memory::data_type::s8:
//       std::cout << "data type:s8" << std::endl;
//       break;
//     case memory::data_type::u8:
//       std::cout << "data type:u8" << std::endl;
//       break;
//     default:
//       std::cout << "unknown dtype" << std::endl;
//   };
// }

// static void dump_md(const char* str, memory::desc md) {
//   printf("%s\n", str);

//   print_vec_xpu("\tdims : ", md.get_dims().data(), md.get_ndims());

//   print_vec_xpu("\tpdims: ", md.get_padded_dims().data(), md.get_ndims());

//   print_vec_xpu("\toffs : ", md.get_padded_offsets().data(), md.get_ndims());

//   dump_md_data_type(md.get_data_type());

//   print_vec_xpu("\tstrs : ", md.get_strides().data(), md.get_strides().size());

//   printf("\t\tnblks : %d\n", md.get_inner_nblks());

//   print_vec_xpu(
//       "\t\tidxs  : ", md.get_inner_idxs().data(), md.get_inner_nblks());

//   print_vec_xpu(
//       "\t\tblks  : ", md.get_inner_blks().data(), md.get_inner_nblks());
// }

// static inline bool requires_runtime_zp(const Tensor& src) {
//   TORCH_CHECK(src.is_quantized(), "Only qtensor needs runtime zero_point")
//   TORCH_CHECK(
//       src.qscheme() == kPerTensorAffine,
//       "Only per tensor quantization has non-zero zp")
//   // See [Note: Use symmetric quant implementation when zp is 0]
//   return (src.q_zero_point() != 0);
// }

} // namespace oneDNN
} // namespace xpu
