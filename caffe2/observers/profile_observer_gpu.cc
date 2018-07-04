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

#include "caffe2/core/logging.h"
#include "profile_observer.h"

namespace caffe2 {

void ProfileOperatorObserver::Dump() const {
  static std::mutex loggingMutex;
  std::lock_guard<std::mutex> lock(loggingMutex);

  LOG(INFO) << "--------- Starting operator " << subject_->debug_def().type()
            << " op#" << getId() << " ---------";
  for (int i = 0; i < subject_->InputSize(); ++i) {
    if (subject_->InputIsType<TensorCPU>(i)) {
      const auto& tensor = subject_->Input<TensorCPU>(i);
      const auto& name = subject_->debug_def().input(i);
      TensorPrinter printer(name);
      LOG(INFO) << "Input " << i << ": " << printer.MetaStr(tensor);
    } else if (subject_->InputIsType<TensorCUDA>(i)) {
      const auto& tensor = subject_->Input<TensorCUDA>(i);
      const auto& name = subject_->debug_def().input(i);
      TensorPrinter printer(name);
      LOG(INFO) << "Input " << i << ": " << printer.MetaStr(tensor);
    }
  }

  int a = 0;
  for (const auto& arg : subject_->debug_def().arg()) {
    LOG(INFO) << "Argument " << a << ": " << arg.ShortDebugString();
    ++a;
  }

  for (int o = 0; o < subject_->OutputSize(); ++o) {
    if (subject_->OutputIsType<TensorCPU>(o)) {
      auto* tensor = subject_->Output<TensorCPU>(o);
      const auto& name = subject_->debug_def().output(o);
      TensorPrinter printer(name);
      LOG(INFO) << "Output " << o << ": " << printer.MetaStr(*tensor);
    } else if (subject_->OutputIsType<TensorCUDA>(o)) {
      auto* tensor = subject_->Output<TensorCUDA>(o);
      const auto& name = subject_->debug_def().output(o);
      TensorPrinter printer(name);
      LOG(INFO) << "Output " << o << ": " << printer.MetaStr(*tensor);
    }
  }

  LOG(INFO) << "--------- Finished operator " << subject_->debug_def().type()
            << " in " << run_time_ << " ms ---------";
}

void ProfileOperatorObserver::Start() {
  auto cudaOp = dynamic_cast_if_rtti<const Operator<CUDAContext>*>(subject_);
  if (cudaOp) {
    auto context = cudaOp->getContext();
    int device;
    cudaGetDevice(&device);

    cudaSetDevice(context->cuda_gpu_id());
    cudaEventCreate(&start_);
    cudaEventRecord(start_, context->cuda_stream());

    cudaSetDevice(device);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      CAFFE_THROW("Encountered CUDA error Start: ", cudaGetErrorString(error));
    }
  } else {
    start_time_ = timer_.MilliSeconds();
  }
}

void ProfileOperatorObserver::Stop() {
  auto cudaOp = dynamic_cast_if_rtti<const Operator<CUDAContext>*>(subject_);
  if (cudaOp) {
    auto context = cudaOp->getContext();
    int device;
    cudaGetDevice(&device);

    cudaSetDevice(context->cuda_gpu_id());
    cudaEventCreate(&stop_);
    cudaEventRecord(stop_, context->cuda_stream());
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&run_time_, start_, stop_);
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);

    cudaSetDevice(device);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      CAFFE_THROW("Encountered CUDA error Stop: ", cudaGetErrorString(error));
    }
  } else {
    run_time_ = timer_.MilliSeconds() - start_time_;
  }

  Dump();
}

std::unique_ptr<ObserverBase<OperatorBase>> ProfileOperatorObserver::rnnCopy(
    OperatorBase* subject,
    int rnn_order) const {
  return std::unique_ptr<ObserverBase<OperatorBase>>(
      new ProfileOperatorObserver(
          subject, netObserver_, net_position_, rnn_order));
}
} // namespace caffe2
