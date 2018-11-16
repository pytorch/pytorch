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
    if (subject_->InputIsTensorType(i, CPU)) {
      const auto& tensor = subject_->Input<Tensor>(i, CPU);
      const auto& name = subject_->debug_def().input(i);
      TensorPrinter printer(name);
      LOG(INFO) << "Input " << i << ": " << printer.MetaStr(tensor);
    } else if (subject_->InputIsTensorType(i, CUDA)) {
      const auto& tensor = subject_->Input<Tensor>(i, CUDA);
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
    if (subject_->OutputIsTensorType(o, CPU)) {
      auto* tensor = subject_->Output<Tensor>(o, CPU);
      const auto& name = subject_->debug_def().output(o);
      TensorPrinter printer(name);
      LOG(INFO) << "Output " << o << ": " << printer.MetaStr(*tensor);
    } else if (subject_->OutputIsTensorType(o, CUDA)) {
      auto* tensor = subject_->Output<Tensor>(o, CUDA);
      const auto& name = subject_->debug_def().output(o);
      TensorPrinter printer(name);
      LOG(INFO) << "Output " << o << ": " << printer.MetaStr(*tensor);
    }
  }
  LOG(INFO) << "Cost (flops, bytes_read, bytes_written, op_type):";
  LOG(INFO) << std::setw(15) << std::setfill(' ') << cost_.flops << " "
            << cost_.bytes_read << " " << cost_.bytes_written << " "
            << subject_->debug_def().type();
  LOG(INFO) << "--------- Finished operator " << subject_->debug_def().type()
            << " in " << run_time_ << " ms ---------";
}

void ProfileOperatorObserver::Start() {
  auto cudaOp = dynamic_cast_if_rtti<const Operator<CUDAContext>*>(subject_);
  if (cudaOp) {
    auto context = cudaOp->getContext();
    int device;
    cudaGetDevice(&device);

    cudaSetDevice(context->device_id());
    cudaEventCreate(&start_);
    cudaEventRecord(start_, context->cuda_stream());

    cudaSetDevice(device);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      CAFFE_THROW("Encountered CUDA error Start: ", cudaGetErrorString(error));
    }
  } else {
    start_time_ = timer_.MilliSeconds();

    cost_ = getOpCost();
    updateDetailedStat(cost_);
  }
}

void ProfileOperatorObserver::Stop() {
  auto cudaOp = dynamic_cast_if_rtti<const Operator<CUDAContext>*>(subject_);
  if (cudaOp) {
    auto context = cudaOp->getContext();
    int device;
    cudaGetDevice(&device);

    cudaSetDevice(context->device_id());
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
          subject, stat_, netObserver_, net_position_, rnn_order));
}

ProfileObserver::~ProfileObserver() {
  static std::mutex loggingMutex;
  std::lock_guard<std::mutex> lock(loggingMutex);

  CaffeMap<string, OpSchema::Cost> cost_per_op_type = getAggregatedOpTypeCost();
  // sort by decreasing flops.
  std::vector<std::pair<std::string, OpSchema::Cost>> cost_per_op_type_vec(
      cost_per_op_type.begin(), cost_per_op_type.end());
  std::sort(
      cost_per_op_type_vec.begin(),
      cost_per_op_type_vec.end(),
      [](const std::pair<std::string, OpSchema::Cost>& left,
         const std::pair<std::string, OpSchema::Cost>& right) {
        return left.second.flops > right.second.flops;
      });
  LOG(INFO) << "================ Detailed stats for net " << net_name_
            << " ================";
  LOG(INFO) << "Aggregated Cost (flops, bytes_read, bytes_written, op_type):";
  for (const auto& item : cost_per_op_type_vec) {
    LOG(INFO) << std::setw(15) << std::setfill(' ') << item.second.flops << " "
              << item.second.bytes_read << " " << item.second.bytes_written
              << " " << item.first;
  }
}

} // namespace caffe2
