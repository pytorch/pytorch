#include "ATen/core/Generator.h"

namespace at {

/*
* Internal Generator API
*/
namespace detail {

// Global generator state and constants
static std::once_flag cpu_device_flag;
static std::unique_ptr<Generator> default_gen_cpu;

static int64_t num_gpus = -1;
static std::once_flag num_gpu_init_flag;
static std::deque<std::once_flag> cuda_device_flags;
static std::vector<std::unique_ptr<Generator>> default_gens_cuda;

/* 
* Populates global values and creates a generator for CPU.
* Note: the generator on CPU is a 64 bit Mersenne Twister Engine.
* Warning: this function must only be called once!
*/
static void initGlobalCPUGeneratorState(){
  GeneratorState default_gen_state_cpu;
  default_gen_state_cpu.philox_offset_per_thread = 0;
  default_gen_state_cpu.device = -1;
  default_gen_state_cpu.device_type = at::kCPU;
  default_gen_state_cpu.current_seed = 67280421310721;
  std::seed_seq seq({67280421310721});
  default_gen_state_cpu.cpu_engine = std::mt19937_64(seq);
  default_gen_cpu = c10::guts::make_unique<Generator>(default_gen_state_cpu);
}

/* 
* Populates the global variables related to CUDA generators
* Warning: this function must only be called once!
*/
#if !C10_MOBILE
static void initCUDAGenVector(){
  num_gpus = at::detail::getCUDAHooks().getNumGPUs();
  cuda_device_flags.resize(num_gpus);
  default_gens_cuda.resize(num_gpus);
}
#endif

/* 
* Populates global values and creates a generator for CUDA
* Note: the engine in a CUDA generator is instantiated inside
* kernel and here we are only setting up the state for that
* engine
* Warning: this function must only be called once!
*/
#if !C10_MOBILE
static void initGlobalCUDAGeneratorState(int64_t device = -1){
  // Switches to the requested device so engines are properly associated
  // with it.
  GeneratorState default_gen_state_cuda;
  default_gen_state_cuda.philox_offset_per_thread = 0;
  default_gen_state_cuda.device = device;
  default_gen_state_cuda.device_type = at::kCUDA;
  default_gen_state_cuda.current_seed = 67280421310721; // keep cpu/cuda default seed same;
  default_gens_cuda[device] = c10::guts::make_unique<Generator>(default_gen_state_cuda);
}
#endif

/*
* Gets the default generators. Lazily creates one if
* there is none.
*/
Generator& getDefaultGenerator(DeviceType device_type, int64_t device) {
  if(device_type == kCPU){
    std::call_once(cpu_device_flag, initGlobalCPUGeneratorState);
    return *default_gen_cpu;
  }else if(device_type == kCUDA){
    #if !C10_MOBILE
      std::call_once(num_gpu_init_flag, initCUDAGenVector);
      if (device == -1) device = at::detail::getCUDAHooks().current_device();
      AT_ASSERT(device >= 0 && device < num_gpus);
      std::call_once(cuda_device_flags[device], initGlobalCUDAGeneratorState, device);
      return *default_gens_cuda[device];
    #else
      AT_ERROR(DeviceTypeName(device_type), " backend type not available or is not enabled.");
    #endif
  }else{ 
    AT_ERROR(DeviceTypeName(device_type), " backend type not available or is not enabled.");
  }
}

/*
* Creates a GeneratorState instance. 
*/
GeneratorState createGenerator(DeviceType device_type, int64_t device) {
  if(device_type == kCUDA){
    #if !C10_MOBILE
      std::call_once(num_gpu_init_flag, initCUDAGenVector);
      if (device == -1) device = at::detail::getCUDAHooks().current_device();
      AT_ASSERT(device >= 0 && device < num_gpus);
    #else
      AT_ERROR(DeviceTypeName(device_type), " backend type not available or is not enabled.");
    #endif
  }
  GeneratorState new_gen_state;
  new_gen_state.philox_offset_per_thread = 0;
  new_gen_state.device = device;
  new_gen_state.device_type = device_type;
  new_gen_state.current_seed = 67280421310721;
  if(device_type == kCPU){
    std::seed_seq seq({67280421310721});
    new_gen_state.cpu_engine = std::mt19937_64(seq);
  }
  return new_gen_state;
}

/*
* Utility function used in tensor implementations, which
* supplies the default generator to tensors, if an input generator
* is not supplied
*/
Generator* checkGeneratorWithDefault(Generator* expr, Generator* defaultValue) {
  if(!expr)
    return defaultValue;
  return expr;
}

} // namespace detail

/*
* Generator class implementation
*/

/* 
* Gets the generator state
*/
GeneratorState* Generator::getState() {
  std::lock_guard<std::mutex> lock(this->mutex);
  return this->state_.get(); 
}

/* 
* Sets the generator state. Calls state's assign constructor
*/ 
void Generator::setState(GeneratorState* state_in) {
  std::lock_guard<std::mutex> lock(this->mutex);
  if (state_in->device_type != this->state_->device_type) {
    AT_ERROR("Invalid state used for setState() function.");
  }
  *this->state_ = *state_in;
}

/* 
* Returns the current seed of the generator
*/
uint64_t Generator::getCurrentSeed() {
  std::lock_guard<std::mutex> lock(this->mutex);
  return this->state_->current_seed;
}

/* 
* Manually seeds the engine with the seed input
*/
void Generator::setCurrentSeed(uint64_t seed) {
  std::lock_guard<std::mutex> lock(this->mutex);
  this->state_->current_seed = seed;
  this->state_->philox_offset_per_thread = 0;
  if(this->state_->device_type == at::kCPU) {
    // Check this out: http://www.pcg-random.org/posts/cpp-seeding-surprises.html
    std::seed_seq seq({seed});
    this->state_->cpu_engine.seed(seq);
  }
}

/* 
* Gets the CPU engine. Throws error for other engines
*/
std::mt19937_64& Generator::getCPUEngine() {
  std::lock_guard<std::mutex> lock(this->mutex);
  if(this->state_->device_type != at::kCPU) {
    AT_ERROR("getCPUEngine() function called for this Generator. it is only valid in CPU Generator");
  }
  return this->state_->cpu_engine;
}

/* 
* Sets the CPU engine. Throws error for other engines
*/
void Generator::setCPUEngine(std::mt19937_64 engine) {
  std::lock_guard<std::mutex> lock(this->mutex);
  if(this->state_->device_type != at::kCPU) {
    AT_ERROR("setCPUEngine(std::mt19937_64 engine) Invalid function called for this Generator. It is only valid in CPU Generator");
  }
  this->state_->cpu_engine = std::mt19937_64(engine);
}

/* 
* Gets a 64 bit random number.
* Throws error for other engines
*/
uint64_t Generator::random64() {
  std::lock_guard<std::mutex> lock(this->mutex);
  if(this->state_->device_type != at::kCPU) {
    AT_ERROR("random64() function called for this Generator. It is only valid in CPU Generator");
  }
  return this->state_->cpu_engine();
}

/* 
* Increments the philox offset of CUDA Generator when called in a kernel launch 
* returns the current seed and the old offset (before increment) as a std::pair 
* for the kernel to use
*
* Throws error when used with non-CUDA generator
* Inputs:
*   total_elements: should be *.numel() call on a Tensor
*   grid_size:         Total number of blocks used in a kernel
*   block_size:        Total number of threads used in a block
*   num_engine_calls:  Total number of operator() calls made in a kernel
*                      this number should include loop unrolling number in a thread
*                      For instance, if loop unrolling is 4, and number of engine()
*                      calls is 2, num_engine_calls should be 4*2=8
*
* How is incrementPhiloxOffset Calculated?
* This function takes care of the calculation AFTER Step 1. It is the
* kernel writer's responsibility to provide step 1, before using this function.
*
* Step 1: find how many elements per thread your kernel will produce?
*   - To do that, you need the grid size, block size and number of elements
*   - Define a block size, b. For instance, block size = 16
*   - Get you output tensor size, t. For instance a.numel() = 1024
*   - Grid size, g = t / b
*   - Ask, is grid size greater than what is available? Find that out by:
*     - unsigned int blocks_per_sm = 
*        at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
*     - g_new = std::min(
*                 (unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, 
*                  grid.x);
*   - Now count the number of operator() calls you make to PhiloxEngine in your kernel.
*     Let this number be m.
*
* Step 2: This is what Generator::incrementPhiloxOffset implements. You DON'T have to do this
*         step if you use this function.
*   - Now that you have the actual grid_size, block_size and number of elements,
*     find out how many elements per thread your kernel will actually produce, by:
*     Number of Elements per Thread, n = t / (g_new * b)
*   - Therefore, the total number of randoms we need in a thread is (m x n).
*   - Hence, since a PhiloxEngine is launched in parallel and assigned a unique thread index 
*     (i.e. each thread is a subsequence of the engine and has a max 2^64 random values available
*     per thread), we want to ensure that we have the right spacing/offset in that subsequence, when
*     a random number is requested. In other words, if we don't have the right spacing, we may
*     end up reusing some random numbers, which is undesired. The offset variable in PhiloxEngine 
*     decides how many 128-bit random numbers to skip (i.e. how many groups of 4, 32-bit numbers to skip) 
*     and hence makes sure that random numbers are not reused in the subsequences that the blocks use.
*   - Therefore, the increment formula for the following function is: 
*       ((total_elements / (grid_size * block_size)) * num_engine_calls ) / 4
*   - Note that we are dividing by 4 since, the philox engine produces 4 values at a time. 
*     So if we are skipping {1,2,3,4} values, increment is 1, if skipping {5,6,7,8} increment is 2 and
*     so on.
*   
*   - Here is a visual of how this function works (if it explains better), when increment is 1:
*   - a subsequent kernel launch starts from the incremented offset value
*   thread 0 randoms            thread 1 randoms             thread 2 randoms
*   [2,1,3,7,5,8...10]          [11,2,3,5,...99]             [187,298,398,885,...236]
*   ^        ^                    ^       ^                     ^             ^
*   |        | incremented offset |       | incremented offset  |             | incremented offset
*   |                             |                             |
*   current offset              current offset                current offset
*/
std::pair<uint64_t, uint64_t> Generator::incrementPhiloxOffset(uint64_t total_elements, 
                                                         uint64_t grid_size,
                                                         uint64_t block_size,
                                                         uint64_t num_engine_calls) {
  std::lock_guard<std::mutex> lock(this->mutex);
  if(this->state_->device_type != at::kCUDA) {
    AT_ERROR("incrementPhiloxOffset() function called for this Generator. It is only valid in CUDA Generator");
  }
  // See the note above for the break down of this formula. Doing an ceil integer division
  // on the single formula.
  uint64_t numel_per_thread = (total_elements - 1)/(block_size * grid_size * 4) + 1;
  uint64_t increment = numel_per_thread * num_engine_calls;
  // we do a fetch_add such that, the philox_offset_per_thread is in a running state.
  // i.e. if you want to fork the philox RNG, you should create a new generator instance
  // philox_offset_per_thread is the only way we are exposing the state of the engine from the cuda
  // generator.
  // Each kernel using philox has to sensibly increment offset for future users of philox. So it gets the 
  // "old" value for itself (before add), and tells subsequent users which offset they should use, 
  // since only the kernel knows how many randoms it intends to generate.
  uint64_t offset = this->state_->philox_offset_per_thread;
  this->state_->philox_offset_per_thread += increment;
  return std::make_pair(this->state_->current_seed, offset);
}

} // namespace at
