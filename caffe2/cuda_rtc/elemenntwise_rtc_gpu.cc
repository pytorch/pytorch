#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"
#include "caffe2/cuda_rtc/common_rtc.h"

namespace caffe2 {
namespace {
class ElementwiseRTCFunction
    : public CudaRTCFunction<ElementwiseRTCFunction> {
 public:
  ElementwiseRTCFunction() : CudaRTCFunction(), name_(GetUniqueName()) {}

  template <typename... Args>
  string KernelName(Args... /*args*/) {
    return name_;
  }

  template <typename... Args>
  string GetSource(Args... args);

 private:
  string name_;
};

template<>
string ElementwiseRTCFunction::GetSource(
    int input_size, int output_size,
    const string command_string) {
  std::stringstream ss;
  ss << "extern \"C\" __global__ void " << name_ <<
        "(const size_t nthreads, \n";
  // Insert the parameter list.
  int remain_params = input_size + output_size;
  for (int i = 0; i < input_size; ++i) {
    ss << "const float* in" << i
       << ((remain_params--) ? ", \n" : "");
  }
  for (int i = 0; i < output_size; ++i) {
    ss << "float* out" << i
       << ((remain_params--) ? ", \n" : "");
  }
  ss << ") {\n"
        "for (int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "index < nthreads; index += blockDim.x * gridDim.x) {\n"
     << command_string << "\n"
     << "}\n}";
  return ss.str();
}
}  // namespace

/**
 * A GPU operator that can generate limited elementwise operations.
 *
 * ElementwiseRTCOp allows one to do a simple and limited thing: it takes in
 * multiple inputs and multiple outputs, as well as a raw string argument
 * rtc_src. The runtime then generates the following kernel code:
 *
 *   __global__ void kernel_name(const size_t nthreads, ...) {
 *     for(int index = blockIdx.x * blockDim.x + threadIdx.x;
 *         index < nthreads; index += blockDim.x * gridDim.x) {
 *       rtc_src
 *     }
 *   }
 * where the "..." part is auto generated, so one can refer to the input and
 * output as in0, in1, ..., out0, out1... in the rtc_src string.
 *
 * For example, if one wants to do a vector multiplication, one can take two
 * inputs and one outputs, and write rtc_src as
 *     out0[index] = in0[index] * in1[index];
 *
 * This op is currently highly experimental. We do not have a gradient
 * registered for it either.
 */
class ElementwiseRTCOp final : public Operator<CUDAContext> {
 public:
  ElementwiseRTCOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {
    const string src = OperatorBase::GetSingleArgument<string>(
        "rtc_src", "");
    CAFFE_ENFORCE(src.size(), "Op should have a non-zero source code size.");
    func_.Compile(InputSize(), OutputSize(), src);
  }
  ~ElementwiseRTCOp() {}

  bool RunOnDevice() override {
    static_assert(sizeof(void*) == sizeof(size_t),
                  "The argbuffer relies on the assumption that void* and "
                  "size_t have the same size.");
    vector<size_t> argBuffer_vec(InputSize() + OutputSize() + 1);
    size_t* argBuffer = argBuffer_vec.data();
    CAFFE_ENFORCE(
        Input(0).size() < std::numeric_limits<int>::max(),
        "The kernel function currently only supports int index.");
    argBuffer[0] = Input(0).size();
    void** ptr_buffer = reinterpret_cast<void**>(argBuffer + 1);
    for (int i = 0; i < InputSize(); ++i) {
      ptr_buffer[i] = const_cast<float*>(Input(i).data<float>());
    }
    for (int i = 0; i < OutputSize(); ++i) {
      Output(i)->ResizeLike(Input(0));
      ptr_buffer[i + InputSize()] = Output(i)->mutable_data<float>();
    }
    size_t argBufferSize = sizeof(argBuffer);
    void* config[] = {
      CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
      CU_LAUNCH_PARAM_BUFFER_SIZE, &argBufferSize,
      CU_LAUNCH_PARAM_END
    };
    func_.LaunchEx(CAFFE_GET_BLOCKS(Input(0).size()), 1, 1,
                   CAFFE_CUDA_NUM_THREADS, 1, 1,
                   0, context_.cuda_stream(), config);
    return true;
  }

 private:
  ElementwiseRTCFunction func_;
};

namespace {
REGISTER_CUDA_OPERATOR_WITH_ENGINE(ElementwiseRTC, NVRTC, ElementwiseRTCOp);
}

}  // namespace caffe2
