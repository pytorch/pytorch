#include <torch/csrc/jit/fuser/cuda/interface.h>
#include <torch/csrc/jit/fuser/common/management.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <ATen/CUDAGenerator.h>
#include <THC/THC.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/utils/memory.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/shape_analysis.h>

#include <torch/csrc/jit/fuser/cuda/cs_ir/test.h>

// #include "../common/ir.h"
// #include "../common/ir_printer.h"
// #include "../common/expr.h"

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
// See NOTE [ USE OF NVRTC AND DRIVER API ]
static const at::cuda::NVRTC& nvrtc() {
  return at::globalContext().getNVRTC();
}

static int ceilDiv(const int a, const int b) {
  return (a + b - 1) / b;
}

} // namespace

static std::shared_ptr<Graph> normalizeGraphForCache(
    const std::shared_ptr<Graph>& graph) {
  auto result = Canonicalize(graph, /*keep_unique_names=*/false);
  EraseShapeInformation(result);
  return result;
}

struct KernelCache {

  struct KernelEntry {
    int16_t device_;
    int maxBlocks_;
    cudaDeviceProp* prop_;
    CUmodule module_;
    CUfunction function_;
  };

  //TODO: old design {hash table by key of graph, nested hash table by key of
  //      arg_spec} looks reasonable to me. keep it simple for now.
  //      We would need something similar to cover the BailOut cases where shape
  //      is not available -> graph is not unique for different input shapes? Not
  //      quite sure how/whether shape is provided at run-time yet.
  //TODO: move this into "common/manager.h"
  //unordered_map<std::string, FusedKernel> kernel_map_;
  std::unordered_map<std::string, std::unique_ptr<KernelEntry>> kernel_map_;
  std::mutex mutex_;
};

static KernelCache kernel_cache_;

std::vector<bool> canCollapseDimsDown(const std::shared_ptr<c10::TensorType> tensor){
  int64_t ndims = *(tensor->dim());

  //Flags to see if the current dim can be fused with the one after
  //Goes left to right, furthest right doesn't need a flag
  std::vector<bool> canCollapseDown(ndims, true);

  for (int64_t d = 0; d < ndims - 1; d++) {
    int64_t stride = *(tensor->strides()[d]);
    int64_t stride_p_1 = *(tensor->strides()[d+1]);
    int64_t size_p_1 = *(tensor->sizes()[d+1]);

    if( (stride_p_1 * size_p_1 != stride)
	&& !(stride_p_1 == 0 && stride == 0) )
      canCollapseDown[d] = false;

  }

  canCollapseDown[ndims-1] = true;

  return canCollapseDown;
}

// Returns true if the node is added to the fusion group, false o.w.
bool CUDAFusionBackend::isFusible(const Node* const node) {

  int64_t ndims = *(node->inputs()[0]->type()->expect<TensorType>()->dim());
  std::vector< std::vector<bool> > collapse_vecs;


  //Check how we could dimensionally reduce each input
  for(const auto& value : node->inputs())
    if(value->isCompleteTensor()){
      assert(*(value->type()->expect<TensorType>()->dim()) == ndims);
      collapse_vecs.push_back(canCollapseDimsDown(value->type()->expect<TensorType>()));
    }

  //Check how we could dimennsionally reduce each output
  for(const auto& value : node->outputs())
    if(value->isCompleteTensor()){
      assert(*(value->type()->expect<TensorType>()->dim()) == ndims);
      collapse_vecs.push_back(canCollapseDimsDown(value->type()->expect<TensorType>()));
    }

  std::vector<bool> dim_collapse = collapse_vecs[0];

  for(auto it = collapse_vecs.begin() + 1; it!=collapse_vecs.end(); ++it){
    for(int64_t d = 0; d<ndims; d++){
      dim_collapse[d] = dim_collapse[d] && (*it)[d];
    }
  }

  //Contig not the right word here because the tensor:
  //Size(4, 4, 2) stride(16, 4, 2) will be fully
  //collapsable but not contiguous
  bool contig = true;
  for(const auto iscontig : dim_collapse)
    contig = contig && iscontig;

  if(contig)
    std::cout<<"All tensors are contiguous"<<std::endl;

  bool first = true;
  for (auto i = decltype(dim_collapse.size()){0}; i < dim_collapse.size() - 1 ; ++i) {
    if(dim_collapse[i]){
      if(first){
	std::cout<<"Tensors could be collapsed on Dims = ("<<i;
	first = false;
      }else{
	std::cout<<", "<<i;
      }
    }
  }
  if(!first) std::cout<<")"<<std::endl;


  if(node->kind() ==  aten::add){
    std::cout<<"Can fuse node!"<<std::endl;
    return true;
  }

  return false;
}

// dummy kernel
const char *saxpy = "                                           \n\
extern \"C\" __global__                                         \n\
void saxpy(float *x, float *y, float *out, size_t n)            \n\
{                                                               \n\
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;           \n\
  if (tid < n) {                                                \n\
   out[tid] = x[tid] + y[tid];                                  \n\
  }                                                             \n\
}                                                               \n";

int CUDAFusionBackend::fuse(const Node* const node) {
  TORCH_CHECK(isFusible(node), "Trying to fuse nonfusible node!");

  // Copy cat from CPU fuser;
  return getAndIncrementGlobalFusionCounter();
}

void CUDAFusionBackend::compileFusion(Node* fusion) {
  auto graph = normalizeGraphForCache(fusion->g(attr::Subgraph));
  auto repr = graph->toString(false);
  if (kernel_cache_.kernel_map_.count(repr) == 0) {
    std::lock_guard<std::mutex> guard(kernel_cache_.mutex_);

    //TODO: dummy kernel, replace the following with
    //      kernel_cache_[repr] = createFusionKernel(...)
    kernel_cache_.kernel_map_[repr] = torch::make_unique<KernelCache::KernelEntry>();
    {
      auto kernel_entry = kernel_cache_.kernel_map_[repr].get();
      CUcontext pctx = 0;
      AT_CUDA_DRIVER_CHECK(nvrtc().cuCtxGetCurrent(&pctx));
      if (!pctx) {
        std::unique_lock<std::mutex> cudaFreeMutexLock(
            *(c10::cuda::CUDACachingAllocator::getFreeMutex()));
        cudaFree(0);
      }
      const auto prior_device = at::cuda::current_device();
      at::cuda::set_device(kernel_entry->device_);
      const auto prop = at::cuda::getCurrentDeviceProperties();
      kernel_entry->prop_ = prop;
      int major, minor;
      int nvrtc_major, nvrtc_minor;
      AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcVersion(&nvrtc_major, &nvrtc_minor));

      // Short-circuits if NVRTC version too low
      AT_ASSERT(nvrtc_major >= 6);

      std::string kernel_name = "saxpy";
      //std::string kernel_string = "#include<vector>\n";
      std::string kernel_string = "namespace Fuser {\n";
      kernel_string += Fuser::typeinfo + std::string("\n");
      kernel_string += Fuser::saxpy_codegen(kernel_name);
      kernel_string += std::string("\n}");

      std::cout << "---------------------" << std::endl;
      std::cout << kernel_string << std::endl;
      std::cout << "---------------------" << std::endl;
      auto func_name = "Fuser::" + kernel_name + "<" +
          Fuser::getTypeName<Fuser::IO_struct<float>>() + ">";
      std::cout << func_name << std::endl;
      
      // Major and minor is determined by device properties and
      // possibly "downcompiled" to a lower (compatible) compute architecture
      // based on the NVRTC version
      major = prop->major;
      minor = prop->minor;
      nvrtcProgram program;
      AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcCreateProgram(
          &program, kernel_string.c_str(), nullptr, 0, nullptr, nullptr));
      const std::string compute = "--gpu-architecture=compute_" +
          std::to_string(major) + std::to_string(minor);
      const std::vector<const char*> args = {
          "--std=c++11", compute.c_str(), "-default-device"};

      nvrtc().nvrtcAddNameExpression(program, func_name.c_str());
      const auto result =
          nvrtc().nvrtcCompileProgram(program, args.size(), args.data());
      if (result != NVRTC_SUCCESS) {
        size_t logsize;
        nvrtc().nvrtcGetProgramLogSize(program, &logsize);
        std::vector<char> log(logsize);
        nvrtc().nvrtcGetProgramLog(program, log.data());
        std::stringstream cu;
        cu << log.data();
        throw std::runtime_error(cu.str());
      }
      const char *lowered_kernel_name;
      nvrtc().nvrtcGetLoweredName(program, func_name.c_str(), &lowered_kernel_name);

      ResourceGuard holdProgram(
          [&] { AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcDestroyProgram(&program)); });
      AT_CUDA_NVRTC_CHECK(result);
      size_t ptx_size;
      AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTXSize(program, &ptx_size));
      std::vector<char> ptx;
      ptx.resize(ptx_size);
      AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTX(program, ptx.data()));

      AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleLoadData(&(kernel_entry->module_), ptx.data()));
      AT_CUDA_DRIVER_CHECK(
          nvrtc().cuModuleGetFunction(&(kernel_entry->function_), kernel_entry->module_, lowered_kernel_name));
      AT_CUDA_DRIVER_CHECK(nvrtc().cuOccupancyMaxActiveBlocksPerMultiprocessor(
          &kernel_entry->maxBlocks_, kernel_entry->function_, 128, 0));
      kernel_entry->maxBlocks_ *= kernel_entry->prop_->multiProcessorCount;
    }
  }
}

void CUDAFusionBackend::callFusion(
    const Node* const fusion,
    std::vector<at::Tensor>& outputs,
    at::ArrayRef<IValue> inputs) {
  auto graph = normalizeGraphForCache(fusion->g(attr::Subgraph));
  auto repr = graph->toString(false);
  TORCH_CHECK(kernel_cache_.kernel_map_.count(repr) != 0,
      "No compiled engine, something went wrong!");
  auto kernel_entry = kernel_cache_.kernel_map_[repr].get();
  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(kernel_entry->device_);
  auto stream = at::cuda::getCurrentCUDAStream();

  std::vector<void*> arguments;
  size_t numel = outputs[0].numel();
  const auto nBlocks = std::min(kernel_entry->maxBlocks_, ceilDiv(numel, 128));
  // void *operand0 = inputs[0].toTensor().data_ptr();
  // arguments.push_back(&operand0);
  // void *operand1 = inputs[1].toTensor().data_ptr();
  // arguments.push_back(&operand1);
  // void *output = outputs[0].data_ptr();
  // arguments.push_back(&output);
  Fuser::IO_struct<float> operand0;
  auto t = inputs[0].toTensor();
  operand0.data = t.data_ptr<float>();
  for (int i = 0; i < t.dim(); i++) {
    operand0.shapes[i] = t.sizes()[i];
    operand0.strides[i] = t.strides()[i];
  }
  arguments.push_back(&operand0);

  Fuser::IO_struct<float> operand1;
  t = inputs[1].toTensor();
  operand1.data = t.data_ptr<float>();
  for (int i = 0; i < t.dim(); i++) {
    operand1.shapes[i] = t.sizes()[i];
    operand1.strides[i] = t.strides()[i];
  }
  arguments.push_back(&operand1);

  Fuser::IO_struct<float> output;
  output.data = outputs[0].data_ptr<float>();
  for (int i = 0; i < t.dim(); i++) {
    output.shapes[i] = outputs[0].sizes()[i];
    output.strides[i] = outputs[0].strides()[i];
  }
  arguments.push_back(&output);

  AT_CUDA_DRIVER_CHECK(nvrtc().cuLaunchKernel(
      kernel_entry->function_,
      nBlocks,
      1,
      1,
      128,
      1,
      1,
      0,
      stream,
      arguments.data(),
      nullptr));
  // Resets device (see at::DeviceGuard notes above)
  at::cuda::set_device(prior_device);
}

static CUDAFusionBackend cuda_backend;

RegisterFusionBackendEx reg_ex(at::DeviceType::CUDA, &cuda_backend);

}}}}
