/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "benchmark/benchmark.h"

#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"

#define CAFFE2_SKIP_IF_NO_GPU                                      \
  if (!caffe2::NumCudaDevices()) {                                 \
    state.SkipWithError("No CUDA available, skipping benchmark."); \
    return;                                                        \
  }

using namespace caffe2;

static void BM_CUDAContextCreation(benchmark::State& state) {
  CAFFE2_SKIP_IF_NO_GPU;
  volatile CUDAContext context_so_we_do_initialization_work;
  while (state.KeepRunning()) {
    volatile CUDAContext context;
  }
}
BENCHMARK(BM_CUDAContextCreation);

static void BM_CUDAContextStreamAccess(benchmark::State& state) {
  CAFFE2_SKIP_IF_NO_GPU;
  CUDAContext context;
  while (state.KeepRunning()) {
    volatile cudaStream_t stream = context.cuda_stream();
  }
}
BENCHMARK(BM_CUDAContextStreamAccess);

static void BM_cudaGetDevice(benchmark::State& state) {
  CAFFE2_SKIP_IF_NO_GPU;
  int id;
  while (state.KeepRunning()) {
    CUDA_ENFORCE(cudaGetDevice(&id));
  }
}
BENCHMARK(BM_cudaGetDevice);

static void BM_cudaSetDevice(benchmark::State& state) {
  CAFFE2_SKIP_IF_NO_GPU;
  int total = NumCudaDevices();
  int i = 0;
  while (state.KeepRunning()) {
    CUDA_ENFORCE(cudaSetDevice((i++) % total));
  }
}
BENCHMARK(BM_cudaSetDevice);

static void BM_cudaSetAndGetDevice(benchmark::State& state) {
  CAFFE2_SKIP_IF_NO_GPU;
  int total = NumCudaDevices();
  int i = 0;
  int id;
  while (state.KeepRunning()) {
    CUDA_ENFORCE(cudaSetDevice((i++) % total));
    CUDA_ENFORCE(cudaGetDevice(&id));
  }
}
BENCHMARK(BM_cudaSetAndGetDevice);

static void BM_cudaSetSameDevice(benchmark::State& state) {
  CAFFE2_SKIP_IF_NO_GPU;
  while (state.KeepRunning()) {
    CUDA_ENFORCE(cudaSetDevice(0));
  }
}
BENCHMARK(BM_cudaSetSameDevice);

static void BM_cudaStreamCreateSyncDelete(benchmark::State& state) {
  CAFFE2_SKIP_IF_NO_GPU;
  cudaStream_t stream;
  while (state.KeepRunning()) {
    CUDA_ENFORCE(cudaStreamCreate(&stream));
    CUDA_ENFORCE(cudaStreamSynchronize(stream));
    CUDA_ENFORCE(cudaStreamDestroy(stream));
  }
}
BENCHMARK(BM_cudaStreamCreateSyncDelete);

static void BM_cudaStreamSynchronize(benchmark::State& state) {
  CAFFE2_SKIP_IF_NO_GPU;
  cudaStream_t stream;
  CUDA_ENFORCE(cudaStreamCreate(&stream));
  while (state.KeepRunning()) {
    CUDA_ENFORCE(cudaStreamSynchronize(stream));
  }
}
BENCHMARK(BM_cudaStreamSynchronize);

static void BM_cudaEventRecord(benchmark::State& state) {
  CAFFE2_SKIP_IF_NO_GPU;
  cudaStream_t stream;
  cudaEvent_t event;
  CUDA_ENFORCE(cudaStreamCreate(&stream));
  CUDA_ENFORCE(cudaEventCreateWithFlags(
      &event, cudaEventDefault | cudaEventDisableTiming));
  while (state.KeepRunning()) {
    CUDA_ENFORCE(cudaEventRecord(event, stream));
  }
}
BENCHMARK(BM_cudaEventRecord);

static void BM_cudaStreamWaitEventThenStreamSynchronize(
    benchmark::State& state) {
  CAFFE2_SKIP_IF_NO_GPU;
  cudaStream_t stream;
  cudaEvent_t event;
  CUDA_ENFORCE(cudaStreamCreate(&stream));
  CUDA_ENFORCE(cudaEventCreateWithFlags(
      &event, cudaEventDefault | cudaEventDisableTiming));
  CUDA_ENFORCE(cudaEventRecord(event, stream));
  CUDA_ENFORCE(cudaStreamWaitEvent(stream, event, 0));
  CUDA_ENFORCE(cudaStreamSynchronize(stream));
  while (state.KeepRunning()) {
    CUDA_ENFORCE(cudaStreamWaitEvent(stream, event, 0));
    CUDA_ENFORCE(cudaStreamSynchronize(stream));
  }
}
BENCHMARK(BM_cudaStreamWaitEventThenStreamSynchronize);

static void BM_CudaPointerAffinity(benchmark::State& state) {
  CAFFE2_SKIP_IF_NO_GPU;
  Tensor tensor(vector<int64_t>{1, 2, 3, 4}, CUDA);
  float* ptr = tensor.mutable_data<float>();
  while (state.KeepRunning()) {
    volatile int id = GetGPUIDForPointer(ptr);
  }
}
BENCHMARK(BM_CudaPointerAffinity);

namespace {
template <class Context>
class DummyEmptyOp : public Operator<Context> {
 public:
  DummyEmptyOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}

  bool RunOnDevice() final { return true; }
};

REGISTER_CPU_OPERATOR(DummyEmpty, DummyEmptyOp<CPUContext>);
REGISTER_CUDA_OPERATOR(DummyEmpty, DummyEmptyOp<CUDAContext>);
OPERATOR_SCHEMA(DummyEmpty);
}  // namespace

static void BM_OperatorCreationCPU(benchmark::State& state) {
  std::unique_ptr<OperatorBase> op;
  OperatorDef def;
  Workspace ws;
  def.set_type("DummyEmpty");
  def.mutable_device_option()->set_device_type(PROTO_CPU);
  while (state.KeepRunning()) {
    op = CreateOperator(def, &ws);
  }
}
BENCHMARK(BM_OperatorCreationCPU);

static void BM_OperatorCreationCUDA(benchmark::State& state) {
  CAFFE2_SKIP_IF_NO_GPU;
  std::unique_ptr<OperatorBase> op;
  OperatorDef def;
  Workspace ws;
  def.set_type("DummyEmpty");
  def.mutable_device_option()->set_device_type(PROTO_CUDA);
  while (state.KeepRunning()) {
    op = CreateOperator(def, &ws);
  }
}
BENCHMARK(BM_OperatorCreationCUDA);

static void BM_RawAllocDeallocCPU(benchmark::State& state) {
  while (state.KeepRunning()) {
    // Allocating only 1 byte in order to measure the overhead.
    auto data_ptr = GetCPUAllocator()->allocate(1);
    // Deallocated when it's out of scope
  }
}
BENCHMARK(BM_RawAllocDeallocCPU);

static void BM_TensorAllocDeallocCPU(benchmark::State& state) {
  Tensor tensor(CPU);
  // small allocation
  tensor.Resize(32, 32);
  while (state.KeepRunning()) {
    CHECK(tensor.mutable_data<float>());
    tensor.FreeMemory();
  }
}
BENCHMARK(BM_TensorAllocDeallocCPU);

static void BM_TensorAllocDeallocCUDA(benchmark::State& state) {
  CAFFE2_SKIP_IF_NO_GPU;
  Tensor tensor(CUDA);
  // small allocation
  tensor.Resize(32, 32);
  while (state.KeepRunning()) {
    CHECK(tensor.mutable_data<float>());
    tensor.FreeMemory();
  }
}
BENCHMARK(BM_TensorAllocDeallocCUDA);

BENCHMARK_MAIN();
