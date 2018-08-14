#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)

#include "torch/csrc/jit/fusers/cuda/cuda_fusion_function.h"

#include "ATen/ATen.h"
#include "ATen/DeviceGuard.h"
#include "ATen/cuda/CUDAContext.h"
#include "THC/THC.h"
#include "THC/THCGenerator.hpp"

#include "torch/csrc/cuda/cuda_check.h"
#include "torch/csrc/variable_tensor_functions.h"

#include "torch/csrc/jit/assertions.h"

#include "torch/csrc/jit/fusers/cuda/cuda_fuser.h"
#include "torch/csrc/jit/fusers/cuda/tensor_info.h"

#include <sstream>
#include <stdexcept>

THCGenerator* THCRandom_getGenerator(THCState* state);


namespace torch { namespace jit { namespace cudafuser {

static void checkCUDAVersion(const cudaDeviceProp& prop) {
  if ((prop.major >= 6 && CUDA_VERSION < 8000) ||
      (prop.major >= 7 && CUDA_VERSION < 9000)) {
    std::stringstream err_string;
    err_string << "In CompiledCUDAFusionFunction, PyTorch compiled with insufficient CUDA version: "
         << CUDA_VERSION << " for the current GPU device " << prop.name
         << " with device capability " << prop.major << "." << prop.minor;
    throw std::runtime_error(err_string.str());
  }
}

// Tries to compress sizes and strides according to cont. Emits the result t
// c_sizes, c_strides and throws an error on failure (if can't compress)
static void compressContiguous(
  at::IntList sizes
, at::IntList strides
, const std::vector<bool> &cont
, uint32_t* c_sizes
, uint32_t* c_strides) {
  size_t compressed_dims = 0;
  size_t cur = 0;
  size_t ndim = sizes.size();
  while(cur < ndim) {
    size_t total_size = sizes[cur];
    cur++;
    while(cont[cur-1] && cur < ndim) {
      JIT_ASSERT(strides[cur-1] == sizes[cur]*strides[cur]);
      total_size *= sizes[cur];
      cur++;
    }
   // cur starts pointing at the beginning of run to compress
   // cur ends one _after_ the terminating false or end of list.
   // total_size is the size of all dimensions [begin,end)
   // examples:
   // f = not cont.
   // t = cont.
   // x = don't care, including past end of list
   // s = start of cur
   // e = end of cur


   // f x x x
   // s e

   //  t f x x
   //  s   e

   //  t t f x
   //  s     e

    c_sizes[compressed_dims] = total_size;
    c_strides[compressed_dims] = strides[cur-1];
    compressed_dims++;
  }
  JIT_ASSERT(!cont.back() || strides.back() == 1);
}

static int ceilDiv(int a, int b) { return (a + b - 1) / b; }

CUDAFusionFunction::CUDAFusionFunction(
  const std::string& name
, AnnotatedGraph& agraph)
: name{name}, input_desc{agraph.input_desc}, output_desc{agraph.output_desc} {
  at::DeviceGuard device_guard(agraph.device);

  CUDA_ASSERT(cudaGetDeviceProperties(&prop, agraph.device));
  checkCUDAVersion(prop);

  std::stringstream cu;
  auto ret = emitCompilationUnit(cu, name, agraph, true);
  concat_desc = std::move(ret.first);
  has_random = ret.second;
  compilation_unit = cu.str();
  nvrtcProgram program;
  NVRTC_ASSERT(nvrtcCreateProgram(&program, compilation_unit.c_str(), NULL, 0, nullptr, nullptr));

  std::string compute = "--gpu-architecture=compute_" + std::to_string(prop.major) + std::to_string(prop.minor);
  std::vector<const char *> args = {"--std=c++11", compute.c_str(), "-default-device"};
  nvrtcResult result = nvrtcCompileProgram(program, args.size(), args.data());
  if (result == NVRTC_ERROR_COMPILATION) {
    size_t logsize;
    nvrtcGetProgramLogSize(program, &logsize);
    std::vector<char> log(logsize);
    nvrtcGetProgramLog(program, log.data());
    cu << log.data();
    throw std::runtime_error(cu.str());
  }
  ResourceGuard holdProgram([&] {
    NVRTC_ASSERT(nvrtcDestroyProgram(&program));
  });
  NVRTC_ASSERT(result);

  size_t ptx_size;
  NVRTC_ASSERT(nvrtcGetPTXSize(program, &ptx_size));
  ptx.resize(ptx_size);
  NVRTC_ASSERT(nvrtcGetPTX(program, ptx.data()));

  CU_ASSERT(cuModuleLoadData(&module, ptx.data()));
  CU_ASSERT(cuModuleGetFunction(&function, module, name.c_str()));

  CU_ASSERT(cuOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxBlocks, function, 128, 0));
  maxBlocks *= prop.multiProcessorCount;
}

void CUDAFusionFunction::launch_with_tensors(
  at::ArrayRef<at::Tensor> inputs
, at::ArrayRef<at::Tensor> outputs) {
  at::DeviceGuard device_guard(inputs);
  JIT_ASSERT(inputs.size() == input_desc.size());
  JIT_ASSERT(outputs.size() == output_desc.size());
  size_t flat_outputs_size = 0;
  for(auto& c : concat_desc)
    flat_outputs_size += c.nSubtensors;
  // XXX: this code assumes that inputs are 32-bit addressable
  // XXX: this code assumes that all inputs are of the same size
  JIT_ASSERT(inputs[0].numel() <= std::numeric_limits<uint32_t>::max());
  uint32_t numel = inputs[0].numel();
  at::IntList map_size = inputs[0].sizes();
  // Compute the storage needed to store TensorInfo structs for inputs and outputs.
  size_t uncompressedDim = input_desc.at(0).contiguity.size();
  size_t maxPossibleTensorInfoSize = sizeof(TensorInfo) + 2 * sizeof(uint32_t) * uncompressedDim;
  size_t maxPossibleBufferSize = maxPossibleTensorInfoSize * (inputs.size() + flat_outputs_size);
  std::vector<char> buffer(maxPossibleBufferSize);
  char* buffer_next = buffer.data();
  // A vector of arguments to the kernel. It's (numel, *input_descs, *output_descs)
  std::vector<void*> arguments;
  arguments.reserve(3 + inputs.size() + flat_outputs_size);
  // Asserts that t's dims can be compressed in the same way as in desc
  // (that's what the kernel assumes), and appends it to the arguments vector.
  auto addTensorInfo = [&](TensorDesc& desc, const at::Tensor& t) {
    size_t nDim = desc.nDim(); // NOTE: this is the compressed dim
    JIT_ASSERT(nDim <= uncompressedDim); // We'd overflow the space otherwise
    auto ti = reinterpret_cast<TensorInfo*>(buffer_next);
    ti->data = t.data_ptr();
    compressContiguous(t.sizes(), t.strides(), desc.contiguity, ti->sizes(nDim), ti->strides(nDim));
    buffer_next += maxPossibleTensorInfoSize;
    arguments.push_back(ti);
  };
  arguments.push_back(&numel);
  for (size_t i = 0; i < input_desc.size(); ++i)
    addTensorInfo(input_desc[i], inputs[i]);
  for (size_t i = 0; i < output_desc.size(); ++i) {
    auto& c = concat_desc[i];
    at::Tensor o = outputs[i];
    if(c.nSubtensors == 1) {
      o.resize_(map_size);
      addTensorInfo(output_desc[i], outputs[i]);
    } else {
      size_t small_size = map_size[c.dim];
      std::vector<int64_t> concat_size(map_size.begin(), map_size.end());
      concat_size[c.dim] = small_size * c.nSubtensors;
      o.resize_(concat_size);
      size_t offset = 0;
      for(size_t j = 0; j < c.nSubtensors; ++j) {
        // because the concatenated_output stays live, the underlying data
        // in this view remains live through the end of this function
        // so there is not need to hold onto this tensor
        auto view = o.narrow(c.dim, offset, small_size);
        addTensorInfo(*c.subtensorDesc, view);
        offset += small_size;
      }
    }
  }

  // If the kernel call contains a random op, we need to pass in random seeds as
  // well.
  if (has_random) {
    auto gen_ = THCRandom_getGenerator(at::globalContext().getTHCState());
    uint64_t offset =
        gen_->state.philox_seed_offset.fetch_add(this->get_rand_offset(numel));
    arguments.push_back(&gen_->state.initial_seed);
    arguments.push_back(&offset);
  }

  launch_raw(numel, arguments.data());
}

void CUDAFusionFunction::launch(
  at::ArrayRef<at::Tensor> inputs
, std::vector<at::Tensor>& outputs) {
  at::DeviceGuard guard(inputs.back());
  outputs.clear();
  outputs.reserve(outputDescriptors().size());
  for (auto& od : outputDescriptors()) {
    outputs.push_back(torch::getType(at::kCUDA, od.scalar_type).tensor());
  }

  launch_with_tensors(inputs, outputs);
}

uint64_t CUDAFusionFunction::get_rand_offset(uint32_t numel) {
  int numBlocks = std::min(maxBlocks, ceilDiv(numel, blockSize));
  return 4 * (ceil(numel/(4 * blockSize * numBlocks)) + 1);
}

void CUDAFusionFunction::launch_raw(uint32_t numel, void** arguments) {
  int numBlocks = std::min(maxBlocks, ceilDiv(numel, blockSize));

  //std::cout << "maxBlocks = " << maxBlocks << " needed blocks: " << ceilDiv(numel,blockSize)
  //          << " numblocks =  " << numBlocks;

  // it is possible that this is the first cuda call on this thread
  // so make sure we initialize the Driver API's context
  // cudaFree(0) accomplishes this.
  CUcontext pctx = 0;
  CU_ASSERT(cuCtxGetCurrent(&pctx));

  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
    *(THCCachingAllocator_getCudaFreeMutex()));
    cudaFree(0);
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  CU_ASSERT(cuLaunchKernel(
    function
  , numBlocks, 1, 1
  , blockSize, 1, 1
  , 0
  , stream
  , arguments
  , nullptr));
}

} // namespace cudafuser
} // namespace jit
} // namespace torch

#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)