#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/mps/IndexKernels.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/maximum.h>
#include <ATen/ops/minimum.h>
#endif

namespace at::native {
namespace mps {

enum class BinaryKernelType {
  Scalar,
  LHS_Scalar,
  RHS_Scalar,
  Tensor,
  Strided_LHS_Scalar,
  Strided_RHS_Scalar,
  Strided_Tensor
};

static char BINARY_OP_TEMPLATE_TENSOR[] = R"METAL_BINARY(
kernel void {3}_kernel(uint tid                   [[thread_position_in_grid]],
                       const device {1} * input   [[buffer(0)]],
                       const device {2} * other   [[buffer(1)]],
                       device       {0} * output  [[buffer(2)]]) {{
  output[tid] = ({5})input[tid] {4} ({5})other[tid];
}}
)METAL_BINARY";

static char BINARY_OP_TEMPLATE_STRIDED_TENSOR[] = GET_IDX_TEMPLATE
    R"METAL_BINARY(
kernel void {3}_kernel_strided(uint tid [[thread_position_in_grid]],
                       const device void     * input_           [[buffer(0)]],
                       const device void     * other_           [[buffer(1)]],
                       device       void     * output_          [[buffer(2)]],
                       constant     uint     * iter_shape       [[buffer(3)]],
                       constant     uint     & num_dimensions   [[buffer(4)]],
                       constant packed_uint3 * strides          [[buffer(5)]]) {{
  uint3 offsets = get_idx(tid, iter_shape, num_dimensions, strides);

  device {0}* output  = (device {0}*)((device uint8_t*)output_  + offsets.x);
  const device {1}* input = (const device {1}*)((const device uint8_t*)input_ + offsets.y);
  const device {2}* other = (const device {2}*)((const device uint8_t*)other_ + offsets.z);

  *output = ({5})*input {4} ({5})*other;
}}
)METAL_BINARY";

static char BINARY_OP_TEMPLATE_LHS_SCALAR[] = R"METAL_BINARY(
kernel void {3}_kernel_scalar_lhs(uint tid  [[thread_position_in_grid]],
                              const device {1} & input   [[buffer(0)]],
                              const device {2} * other   [[buffer(1)]],
                              device       {0} * output  [[buffer(2)]]) {{
  output[tid] = ({5})input {4} ({5})other[tid];
}}
)METAL_BINARY";

static char BINARY_OP_TEMPLATE_RHS_SCALAR[] = R"METAL_BINARY(
kernel void {3}_kernel_scalar_rhs(uint tid  [[thread_position_in_grid]],
                              const device {1} * input   [[buffer(0)]],
                              const device {2} & other   [[buffer(1)]],
                              device       {0} * output  [[buffer(2)]]) {{
  output[tid] = ({5})input[tid] {4} ({5})other;
}}
)METAL_BINARY";

static char BINARY_OP_TEMPLATE_SCALAR[] = R"METAL_BINARY(
kernel void {3}_kernel_scalar(uint tid  [[thread_position_in_grid]],
                              const device {1}  & input  [[buffer(0)]],
                              const device {2}  & other  [[buffer(1)]],
                              device       {0}  & output [[buffer(2)]]) {{
  output = ({5})input {4} ({5})other;
}}
)METAL_BINARY";

static char BINARY_OP_TEMPLATE_STRIDED_RHS_SCALAR[] = GET_IDX_TEMPLATE
    R"METAL_BINARY(
kernel void {3}_kernel_scalar_rhs_strided(uint tid               [[thread_position_in_grid]],
                       const device void     * input_            [[buffer(0)]],
                       const device {2}      & other             [[buffer(1)]],
                       device void           * output_           [[buffer(2)]],
                       constant uint         * iter_shape        [[buffer(3)]],
                       constant uint         & num_dimensions    [[buffer(4)]],
                       constant packed_uint3 * strides           [[buffer(5)]]) {{
  uint3 offsets = get_idx(tid, iter_shape, num_dimensions, strides);

  device {0}* output = (device {0}*)((device uint8_t*)output_ + offsets.x);
  const device {1}* input = (const device {1}*)((const device uint8_t*)input_ + offsets.y);

  *output = ({5})*input {4} ({5})other;
}}
)METAL_BINARY";

static char BINARY_OP_TEMPLATE_STRIDED_LHS_SCALAR[] = GET_IDX_TEMPLATE
    R"METAL_BINARY(
kernel void {3}_kernel_scalar_lhs_strided(uint tid               [[thread_position_in_grid]],
                       const device {1}      & input             [[buffer(0)]],
                       const device void     * other_            [[buffer(1)]],
                       device void           * output_           [[buffer(2)]],
                       constant uint         * iter_shape        [[buffer(3)]],
                       constant uint         & num_dimensions    [[buffer(4)]],
                       constant packed_uint3 * strides           [[buffer(5)]]) {{
  uint3 offsets = get_idx(tid, iter_shape, num_dimensions, strides);

  device {0}* output = (device {0}*)((device uint8_t*)output_ + offsets.x);
  const device {2}* other = (const device {2}*)((const device uint8_t*)other_ + offsets.z);

  *output = ({5})input {4} ({5})*other;
}}
)METAL_BINARY";

static id<MTLLibrary> compileBinaryOpsLibrary(id<MTLDevice> device,
                                              const std::string& t1,
                                              const std::string& t2,
                                              const std::string& t3,
                                              const std::string& common_dtype,
                                              const std::string& op,
                                              const std::string& kernel_operator,
                                              BinaryKernelType binaryKernelType) {
  auto key = op + t1 + t2 + t3 + common_dtype + std::to_string(int(binaryKernelType));
  static std::unordered_map<std::string, id<MTLLibrary>> libMap;
  auto it = libMap.find(key);
  if (it != libMap.end()) {
    return it->second;
  }
  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
  MTLLanguageVersion languageVersion = MTLLanguageVersion2_2;
#if defined(__MAC_13_0)
  if (is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_0_PLUS)) {
    languageVersion = MTLLanguageVersion3_0;
  }
#endif

  [options setLanguageVersion:languageVersion];
  char* str = nil;
  switch (binaryKernelType) {
    case BinaryKernelType::Scalar:
      str = BINARY_OP_TEMPLATE_SCALAR;
      break;
    case BinaryKernelType::LHS_Scalar:
      str = BINARY_OP_TEMPLATE_LHS_SCALAR;
      break;
    case BinaryKernelType::RHS_Scalar:
      str = BINARY_OP_TEMPLATE_RHS_SCALAR;
      break;
    case BinaryKernelType::Tensor:
      str = BINARY_OP_TEMPLATE_TENSOR;
      break;
    case BinaryKernelType::Strided_Tensor:
      str = BINARY_OP_TEMPLATE_STRIDED_TENSOR;
      break;
    case BinaryKernelType::Strided_LHS_Scalar:
      str = BINARY_OP_TEMPLATE_STRIDED_LHS_SCALAR;
      break;
    case BinaryKernelType::Strided_RHS_Scalar:
      str = BINARY_OP_TEMPLATE_STRIDED_RHS_SCALAR;
      break;
    default:
      TORCH_CHECK(false, "Unknown binary template");
  }

  auto rc = [device
      newLibraryWithSource:[NSString
                               stringWithUTF8String:fmt::format(str, t1, t2, t3, op, kernel_operator, common_dtype)
                                                        .c_str()]
                   options:options
                     error:&error];
  TORCH_CHECK(rc != nil && error == nil, "Failed to compile library: ", [[error localizedDescription] UTF8String]);
  libMap[key] = rc;
  return rc;
}

static id<MTLComputePipelineState> getBinaryPSO(id<MTLDevice> device,
                                                const std::string& t1,
                                                const std::string& t2,
                                                const std::string& t3,
                                                const std::string& common_dtype,
                                                const std::string& fname,
                                                const std::string& op,
                                                const std::string& kernel_operator,
                                                BinaryKernelType binaryKernelType) {
  auto key = t1 + t2 + t3 + common_dtype + fname;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cplMap;
  auto it = cplMap.find(key);
  if (it != cplMap.end()) {
    return it->second;
  }
  NSError* error = nil;
  auto library = compileBinaryOpsLibrary(device, t1, t2, t3, common_dtype, op, kernel_operator, binaryKernelType);
  id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:fname.c_str()]];
  TORCH_CHECK(func != nil, "Can't get function ", fname);
  auto rc = [device newComputePipelineStateWithFunction:func error:&error];
  TORCH_CHECK(
      rc != nil && error == nil, "Failed to construct pipeline state: ", [[error localizedDescription] UTF8String]);
  cplMap[key] = rc;
  return rc;
}

static void dispatch_binary_kernel_mps_(TensorIteratorBase& iter,
                                        const std::string& op,
                                        const std::string& kernel_operator) {
  Tensor inputTensor;
  Tensor otherTensor;
  BinaryKernelType type;

  int scalar_pos = 0;
  bool all_scalar = false;
  const Tensor& outputTensor = iter.tensor(0);
  inputTensor = iter.tensor(1);
  otherTensor = iter.tensor(2);

  if (inputTensor.scalar_type() == kDouble) {
    inputTensor = inputTensor.to(iter.common_dtype());
  }
  if (otherTensor.scalar_type() == kDouble) {
    otherTensor = otherTensor.to(iter.common_dtype());
  }

  auto outputDataType = outputTensor.scalar_type();
  auto inputDataType = inputTensor.scalar_type();
  auto otherDataType = otherTensor.scalar_type();
  ScalarType common_dtype = iter.common_dtype();
  if (isIntegralType(common_dtype, true)) {
    // integer inputs must be cast to float, if output is float
    if (isFloatingType(outputDataType)) {
      common_dtype = outputDataType;
      // in boolean comparison ops with signed vs. unsigned integers, we always cast to the unsigned type
    } else if (outputDataType == ScalarType::Bool &&
               (inputDataType == ScalarType::Byte || otherDataType == ScalarType::Byte)) {
      common_dtype = ScalarType::Byte;
    }
  }

  // workaround for bool issues (e.g. bool dtype: true + true in Metal would be 0, but the expected result is still 1 in
  // PyTorch)
  if (outputDataType == kBool && (inputDataType == kByte || otherDataType == kByte)) {
    inputDataType = otherDataType = kByte;
  } else {
    if (inputDataType == kBool) {
      inputDataType = kChar;
    }
    if (otherDataType == kBool) {
      otherDataType = kChar;
    }
  }

  if (iter.tensor(1).numel() == 1 && iter.tensor(2).numel() == 1) {
    all_scalar = true;
  } else if (iter.tensor(1).numel() == 1) {
    scalar_pos = 1;
  } else if (iter.tensor(2).numel() == 1) {
    scalar_pos = 2;
  }

  if (!scalar_pos && !all_scalar) {
    std::vector<Tensor> tmp = expand_outplace({inputTensor, otherTensor});
    inputTensor = tmp[0];

    otherTensor = tmp[1];
  }

  if (inputTensor.numel() == 0 || otherTensor.numel() == 0) {
    return;
  }

  bool allContiguous = false;
  if (inputTensor.is_contiguous() && otherTensor.is_contiguous() && outputTensor.is_contiguous()) {
    allContiguous = true;
  }

  MPSStream* mpsStream = getCurrentMPSStream();
  id<MTLDevice> device = MPSDevice::getInstance()->device();

  id<MTLBuffer> inputBuffer = mps::getMTLBufferStorage(inputTensor);
  id<MTLBuffer> otherBuffer = mps::getMTLBufferStorage(otherTensor);
  id<MTLBuffer> outputBuffer = mps::getMTLBufferStorage(outputTensor);
  uint32_t inputTensorStorage = inputTensor.storage_offset() * inputTensor.element_size();
  uint32_t otherTensorStorage = otherTensor.storage_offset() * otherTensor.element_size();
  mps::MPSScalar scalar;
  if (all_scalar) {
    type = BinaryKernelType::Scalar;
    if (iter.is_cpu_scalar(1)) {
      scalar = mps::getMPSScalar(inputTensor.item(), inputTensor.scalar_type());
      inputBuffer = (id<MTLBuffer>)getIMPSAllocator()->allocScalarBufferWithValue(&scalar.value, scalar.size).get();
      inputTensorStorage = 0;
    }
    if (iter.is_cpu_scalar(2)) {
      scalar = mps::getMPSScalar(otherTensor.item(), otherTensor.scalar_type());
      otherBuffer = (id<MTLBuffer>)getIMPSAllocator()->allocScalarBufferWithValue(&scalar.value, scalar.size).get();
      otherTensorStorage = 0;
    }
  } else if (scalar_pos) {
    if (allContiguous) {
      type = scalar_pos == 1 ? BinaryKernelType::LHS_Scalar : BinaryKernelType::RHS_Scalar;
    } else {
      type = scalar_pos == 1 ? BinaryKernelType::Strided_LHS_Scalar : BinaryKernelType::Strided_RHS_Scalar;
    }

    if (iter.is_cpu_scalar(scalar_pos)) {
      if (scalar_pos == 1) {
        scalar = mps::getMPSScalar(inputTensor.item(), inputTensor.scalar_type());
        inputBuffer = (id<MTLBuffer>)getIMPSAllocator()->allocScalarBufferWithValue(&scalar.value, scalar.size).get();
        inputTensorStorage = 0;
      } else {
        scalar = mps::getMPSScalar(otherTensor.item(), otherTensor.scalar_type());
        otherBuffer = (id<MTLBuffer>)getIMPSAllocator()->allocScalarBufferWithValue(&scalar.value, scalar.size).get();
        otherTensorStorage = 0;
      }
    }
  } else {
    type = allContiguous ? BinaryKernelType::Tensor : BinaryKernelType::Strided_Tensor;
  }

  const uint32_t nDim = iter.ndim();
  constexpr uint32_t nOffsets = 3;

  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      uint32_t numThreads = iter.numel();
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      const IntArrayRef& iterShape = iter.shape();
      std::vector<uint32_t> iterShapeData(iterShape.size());
      std::vector<std::array<uint32_t, nOffsets>> strides(nDim);

      if (!allContiguous) {
        for (const auto i : c10::irange(iterShape.size())) {
          TORCH_CHECK(i <= UINT32_MAX);
          iterShapeData[i] = (uint32_t)(iterShape[i]);
        }

        for (const auto i : c10::irange(nDim)) {
          for (const auto offset : c10::irange(nOffsets)) {
            strides[i][offset] = iter.strides(offset)[i];
          }
        }
      }

      std::string kernel = op;
      kernel += "_kernel";
      if (all_scalar) {
        kernel += "_scalar";
      }
      if (scalar_pos) {
        kernel += "_scalar_";
        if (scalar_pos == 1) {
          kernel += "lhs";
        } else {
          kernel += "rhs";
        }
      }
      if (!allContiguous) {
        kernel += "_strided";
      }

      id<MTLComputePipelineState> binaryPSO = mps::getBinaryPSO(device,
                                                                getMetalScalarType(outputDataType),
                                                                getMetalScalarType(inputDataType),
                                                                getMetalScalarType(otherDataType),
                                                                getMetalScalarType(common_dtype),
                                                                kernel,
                                                                op,
                                                                kernel_operator,
                                                                type);
      [computeEncoder setComputePipelineState:binaryPSO];
      [computeEncoder setBuffer:inputBuffer offset:inputTensorStorage atIndex:0];
      [computeEncoder setBuffer:otherBuffer offset:otherTensorStorage atIndex:1];
      [computeEncoder setBuffer:outputBuffer
                         offset:outputTensor.storage_offset() * outputTensor.element_size()
                        atIndex:2];
      if (!allContiguous) {
        [computeEncoder setBytes:iterShapeData.data() length:sizeof(uint32_t) * iterShape.size() atIndex:3];
        [computeEncoder setBytes:&nDim length:sizeof(uint32_t) atIndex:4];
        [computeEncoder setBytes:strides.data() length:sizeof(uint32_t) * nDim * nOffsets atIndex:5];
      }

      NSUInteger tgSize = binaryPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > numThreads) {
        tgSize = numThreads;
      }

      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    }
  });
}

static void dispatch_binary_kernel_mps(const Tensor& self,
                                       const Tensor& other,
                                       const Tensor& output,
                                       const std::string& op,
                                       const std::string& kernel_operator) {
  TensorIterator iter;
  if (op == "lt" || op == "le" || op == "gt" || op == "ge" || op == "ne" || op == "logical_or" || "logical_and" ||
      op == "eq") {
    iter = TensorIterator::comparison_op(const_cast<Tensor&>(output), self, other);
  } else {
    iter = TensorIterator::borrowing_binary_op(output, self, other);
  }

  dispatch_binary_kernel_mps_(iter, op, kernel_operator);
}

bool getBinaryKernelOperator(const std::string& op_name, std::pair<std::string, std::string>& kernel_operator) {
  static bool macOS13_0_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_0_PLUS);
  if (!macOS13_0_plus) {
    return false;
  }

  static std::unordered_map<std::string, std::pair<std::string, std::string>> opToKernelOperator = {
      {"multiplication", {"mul", "*"}},
      {"div_out_mps:", {"div", "/"}},
      {"add_out_mps:", {"add", "+"}},
      {"sub_out_mps:", {"sub", "-"}},

      // comparison ops
      {"lessThan", {"lt", "<"}},
      {"lessThanOrEqualTo", {"le", "<="}},
      {"greaterThan", {"gt", ">"}},
      {"greaterThanOrEqualTo", {"ge", ">="}},
      {"notEqual", {"ne", "!="}},
      {"logicalOR", {"logical_or", "||"}},
      {"logicalAND", {"logical_and", "&&"}},
      {"equal", {"eq", "=="}},
  };

  auto it = opToKernelOperator.find(op_name);
  if (it == opToKernelOperator.end()) {
    return false;
  }

  kernel_operator = it->second;
  return true;
}

bool dispatchNativeBinaryKernel(const Tensor& self,
                                const Tensor& other,
                                const Tensor& output,
                                const Scalar& alpha,
                                const std::string& op_name) {
  if (alpha.toFloat() == 1.0) {
    std::pair<std::string, std::string> kernel_operator;
    if (getBinaryKernelOperator(op_name, kernel_operator)) {
      dispatch_binary_kernel_mps(self, other, output, kernel_operator.first, kernel_operator.second);
      return true;
    }
  }

  return false;
}

static const char* METAL_BINARY = R"BINARY_METAL(

#include <metal_stdlib>
using namespace metal;

template<typename T>
kernel void fmax(constant void     * input_        [[buffer(0)]],
                  constant void     * other_        [[buffer(1)]],
                  device   void     * out_          [[buffer(2)]],
                  constant uint3    * offsets       [[buffer(3)]],
                  uint tid [[thread_position_in_grid]]) {
  device   T* out   = (device   T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);

  *out = fmax(*input, *other);
}

template<typename T>
kernel void fmin(constant void     * input_        [[buffer(0)]],
                  constant void     * other_        [[buffer(1)]],
                  device   void     * out_          [[buffer(2)]],
                  constant uint3    * offsets       [[buffer(3)]],
                  uint tid [[thread_position_in_grid]]) {
  device   T* out   = (device   T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);

  *out = fmin(*input, *other);
}

template<typename T>
kernel void copysign(constant void     * input_        [[buffer(0)]],
                     constant void     * other_        [[buffer(1)]],
                     device   void     * out_          [[buffer(2)]],
                     constant uint3    * offsets       [[buffer(3)]],
                     uint tid [[thread_position_in_grid]]) {
  device   T* out   = (device   T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);

  *out = copysign(*input, *other);
}

template<typename T>
kernel void copysign_integral(constant void     * input_        [[buffer(0)]],
                     constant void     * other_        [[buffer(1)]],
                     device   void     * out_          [[buffer(2)]],
                     constant uint3    * offsets       [[buffer(3)]],
                     uint tid [[thread_position_in_grid]]) {
  device   float* out = (device float*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);

  *out = copysign(static_cast<float>(*input), static_cast<float>(*other));
}

#define REGISTER_FMAX_OP(DTYPE)                        \
template                                               \
[[host_name("fmax_" #DTYPE)]]                          \
kernel void fmax<DTYPE>(                               \
  constant void     * input_        [[buffer(0)]],     \
  constant void     * other_        [[buffer(1)]],     \
  device   void     * out_          [[buffer(2)]],     \
  constant uint3    * offsets       [[buffer(3)]],     \
  uint tid [[thread_position_in_grid]]);

#define REGISTER_FMIN_OP(DTYPE)                        \
template                                               \
[[host_name("fmin_" #DTYPE)]]                          \
kernel void fmin<DTYPE>(                               \
  constant void     * input_        [[buffer(0)]],     \
  constant void     * other_        [[buffer(1)]],     \
  device   void     * out_          [[buffer(2)]],     \
  constant uint3    * offsets       [[buffer(3)]],     \
  uint tid [[thread_position_in_grid]]);

#define REGISTER_COPYSIGN_OP(DTYPE)                    \
template                                               \
[[host_name("copysign_" #DTYPE)]]                      \
kernel void copysign<DTYPE>(                           \
  constant void     * input_        [[buffer(0)]],     \
  constant void     * other_        [[buffer(1)]],     \
  device   void     * out_          [[buffer(2)]],     \
  constant uint3    * offsets       [[buffer(3)]],     \
  uint tid [[thread_position_in_grid]]);

#define REGISTER_COPYSIGN_INTEGRAL_OP(DTYPE)           \
template                                               \
[[host_name("copysign_" #DTYPE)]]                      \
kernel void copysign_integral<DTYPE>(                  \
  constant void     * input_        [[buffer(0)]],     \
  constant void     * other_        [[buffer(1)]],     \
  device   void     * out_          [[buffer(2)]],     \
  constant uint3    * offsets       [[buffer(3)]],     \
  uint tid [[thread_position_in_grid]]);

REGISTER_FMAX_OP(float);
REGISTER_FMAX_OP(half);
REGISTER_FMIN_OP(float);
REGISTER_FMIN_OP(half);
REGISTER_COPYSIGN_OP(float);
REGISTER_COPYSIGN_OP(half);
REGISTER_COPYSIGN_INTEGRAL_OP(int);
REGISTER_COPYSIGN_INTEGRAL_OP(long);
REGISTER_COPYSIGN_INTEGRAL_OP(short);
REGISTER_COPYSIGN_INTEGRAL_OP(char);
REGISTER_COPYSIGN_INTEGRAL_OP(uchar);
REGISTER_COPYSIGN_INTEGRAL_OP(bool);

)BINARY_METAL";

using namespace mps;

static id<MTLLibrary> compileBinaryOpsLibrary(id<MTLDevice> device) {
  static id<MTLLibrary> binaryLibrary = nil;
  if (binaryLibrary) {
    return binaryLibrary;
  }

  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:MTLLanguageVersion2_3];
  binaryLibrary = [device newLibraryWithSource:[NSString stringWithCString:METAL_BINARY encoding:NSASCIIStringEncoding]
                                       options:options
                                         error:&error];
  TORCH_CHECK(binaryLibrary, "Failed to create metal binary library, error: ", [[error description] UTF8String]);
  return binaryLibrary;
}

static id<MTLComputePipelineState> binaryPipelineState(id<MTLDevice> device, const std::string& kernel) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> psoCache;
  id<MTLComputePipelineState> pso = psoCache[kernel];
  if (pso) {
    return pso;
  }

  NSError* error = nil;
  id<MTLLibrary> binaryLib = compileBinaryOpsLibrary(device);
  id<MTLFunction> binaryFunc = [binaryLib newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
  TORCH_CHECK(binaryFunc, "Failed to create function state object for: ", kernel);
  pso = [device newComputePipelineStateWithFunction:binaryFunc error:&error];
  TORCH_CHECK(pso, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

  psoCache[kernel] = pso;
  return pso;
}

void binary_mps_impl(TensorIteratorBase& iter, const std::string func_name) {
  TORCH_CHECK(iter.common_dtype() != at::kDouble, "float64 is not supported on MPS");

  Tensor input = iter.input(0);
  Tensor other = iter.input(1);
  Tensor out = iter.output();

  id<MTLBuffer> inputBuffer = getMTLBufferStorage(input);
  id<MTLBuffer> otherBuffer = getMTLBufferStorage(other);
  id<MTLBuffer> outputBuffer = getMTLBufferStorage(out);
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  const uint32_t nDim = iter.ndim();
  constexpr uint32_t nOffsets = 3;
  const uint32_t numThreads = iter.numel();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      const IntArrayRef& iterShape = iter.shape();
      std::vector<uint32_t> iterShapeData(iterShape.size());
      std::vector<std::array<uint32_t, nOffsets>> strides(nDim);

      for (const auto i : c10::irange(iterShape.size())) {
        TORCH_CHECK(i <= UINT32_MAX);
        iterShapeData[i] = (uint32_t)(iterShape[i]);
      }

      for (const auto i : c10::irange(nDim)) {
        for (const auto offset : c10::irange(nOffsets)) {
          strides[i][offset] = iter.strides(offset)[i];
        }
      }

      id<MTLComputePipelineState> kernelDataOffsetsPSO =
          MPSDevice::getInstance()->metalIndexingPSO("kernel_index_offsets");
      id<MTLBuffer> kernelDataOffsets = [[device newBufferWithLength:numThreads * sizeof(simd_uint3)
                                                             options:0] autorelease];
      [computeEncoder setComputePipelineState:kernelDataOffsetsPSO];
      [computeEncoder setBytes:strides.data() length:sizeof(uint32_t) * nDim * nOffsets atIndex:0];
      [computeEncoder setBuffer:kernelDataOffsets offset:0 atIndex:1];
      [computeEncoder setBytes:iterShapeData.data() length:sizeof(uint32_t) * iterShape.size() atIndex:2];
      [computeEncoder setBytes:&nDim length:sizeof(uint32_t) atIndex:3];
      [computeEncoder setBytes:&nOffsets length:sizeof(uint32_t) atIndex:4];

      NSUInteger kernelOffsetsTGSize = kernelDataOffsetsPSO.maxTotalThreadsPerThreadgroup;
      if (kernelOffsetsTGSize > numThreads)
        kernelOffsetsTGSize = numThreads;

      MTLSize kernelOffsetsThreadGroupSize = MTLSizeMake(kernelOffsetsTGSize, 1, 1);
      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:kernelOffsetsThreadGroupSize];

      const std::string kernel = func_name + "_" + scalarToMetalTypeString(input.scalar_type());
      id<MTLComputePipelineState> binaryPSO = binaryPipelineState(device, kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(binaryPSO, kernel, {input, other});

      [computeEncoder setComputePipelineState:binaryPSO];
      [computeEncoder setBuffer:inputBuffer offset:input.storage_offset() * input.element_size() atIndex:0];
      [computeEncoder setBuffer:otherBuffer offset:other.storage_offset() * other.element_size() atIndex:1];
      [computeEncoder setBuffer:outputBuffer offset:out.storage_offset() * out.element_size() atIndex:2];
      [computeEncoder setBuffer:kernelDataOffsets offset:0 atIndex:3];

      NSUInteger tgSize = binaryPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > numThreads) {
        tgSize = numThreads;
      }

      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(binaryPSO);
    }
  });
}
} // namespace mps

void fmax_mps_kernel(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    mps::binary_mps_impl(iter, "fmax");
  } else {
    at::maximum_out(const_cast<Tensor&>(iter.output()), iter.input(0), iter.input(1));
  }
}
void fmin_mps_kernel(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    mps::binary_mps_impl(iter, "fmin");
  } else {
    at::minimum_out(const_cast<Tensor&>(iter.output()), iter.input(0), iter.input(1));
  }
}

void copysign_mps_kernel(TensorIteratorBase& iter) {
  mps::binary_mps_impl(iter, "copysign");
}

REGISTER_DISPATCH(fmax_stub, &fmax_mps_kernel);
REGISTER_DISPATCH(fmin_stub, &fmin_mps_kernel);
REGISTER_DISPATCH(copysign_stub, &copysign_mps_kernel);

} // namespace at::native
