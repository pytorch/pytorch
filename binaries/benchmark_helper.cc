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

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#endif

#include <binaries/benchmark_helper.h>
#include "caffe2/core/blob_serialization.h"
#ifdef __CUDA_ARCH__
#include "caffe2/core/context_gpu.h"
#endif
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/utils/bench_utils.h"
#include "caffe2/utils/string_utils.h"
#include <observers/net_observer_reporter_print.h>
#include <observers/observer_config.h>
#include <observers/perf_observer.h>

#if defined(TARGET_OS_MAC) || \
defined(TARGET_OS_IPHONE) || \
defined(TARGET_IPHONE_SIMULATOR)
#include <malloc/malloc.h>
#else
#include <malloc.h>
#endif


void observerConfig() {
  caffe2::ClearGlobalNetObservers();
  caffe2::AddGlobalNetObserverCreator([](caffe2::NetBase* subject) {
    return std::make_unique<caffe2::PerfNetObserver>(subject);
  });
  caffe2::ObserverConfig::setReporter(
      std::make_unique<caffe2::NetObserverReporterPrint>());
}

bool backendCudaSet(const string& backend) {
  bool run_on_gpu = false;
  if (backend == "cuda") {
#ifdef __CUDA_ARCH__
    if (caffe2::HasCudaGPU()) {
      run_on_gpu = true;
    } else {
      CAFFE_THROW("NO GPU support on this host machine");
    }
#else
    CAFFE_THROW("NO GPU support");
#endif
  }
  return run_on_gpu;
}

void setDeviceType(caffe2::NetDef* net_def, caffe2::DeviceType& run_dev) {
  for (int j = 0; j < net_def->op_size(); j++) {
    caffe2::OperatorDef* op = net_def->mutable_op(j);
    op->mutable_device_option()->set_device_type(caffe2::TypeToProto(run_dev));
  }
}

void setOperatorEngine(caffe2::NetDef* net_def, const string& backend) {
  if (backend != "builtin") {
    string engine = backend == "nnpack"
        ? "NNPACK"
        : backend == "eigen" ? "EIGEN"
                             : backend == "mkl" ? "MKLDNN"
                                                : backend == "cuda"
                    ? "CUDA"
                    : backend == "dnnlowp" ? "DNNLOWP"
                                           : backend == "dnnlowp_acc16"
                            ? "DNNLOWP_ACC16"
                            : backend == "default" ? "" : "NONE";
    CAFFE_ENFORCE(engine != "NONE", "Backend is not supported");
    for (int i = 0; i < net_def->op_size(); i++) {
      caffe2::OperatorDef* op_def = net_def->mutable_op(i);
      op_def->set_engine(engine);
    }
  }
}

int loadInput(
    shared_ptr<caffe2::Workspace> workspace,
    const bool run_on_gpu,
    map<string, caffe2::TensorProtos>& tensor_protos_map,
    const string& input,
    const string& input_file,
    const string& input_dims,
    const string& input_type) {
  // How many input blobs are in the inputs
  int blob_num = 1;
  // Load input.
  if (input.size()) {
    vector<string> input_names = caffe2::split(',', input);
    if (input_file.size()) {
      vector<string> input_files = caffe2::split(',', input_file);
      CAFFE_ENFORCE_EQ(
          input_names.size(),
          input_files.size(),
          "Input name and file should have the same number.");
      for (int i = 0; i < input_names.size(); ++i) {
        caffe2::TensorProtos tensor_protos;
        CAFFE_ENFORCE(
            caffe2::ReadProtoFromFile(input_files[i], &tensor_protos));
        workspace->CreateBlob(input_names[i]);
        tensor_protos_map.insert(std::make_pair(input_names[i], tensor_protos));
      }
      // Check that all blobs have the same number of entries
      blob_num = tensor_protos_map[input_names[0]].protos_size();
      for (int i = 1; i < input_names.size(); ++i) {
        int bnum = tensor_protos_map[input_names[i]].protos_size();
        CAFFE_ENFORCE_EQ(
            blob_num,
            bnum,
            "Number of blobs are not the same for all inputs");
      }
    } else if (input_dims.size() || input_type.size()) {
      CAFFE_ENFORCE_GE(
          input_dims.size(),
          0,
          "Input dims must be specified when input tensors are used.");
      CAFFE_ENFORCE_GE(
          input_type.size(),
          0,
          "Input type must be specified when input tensors are used.");

      vector<string> input_dims_list = caffe2::split(';', input_dims);
      CAFFE_ENFORCE_EQ(
          input_names.size(),
          input_dims_list.size(),
          "Input name and dims should have the same number of items.");
      vector<string> input_type_list = caffe2::split(';', input_type);
      CAFFE_ENFORCE_EQ(
          input_names.size(),
          input_type_list.size(),
          "Input name and type should have the same number of items.");
      for (size_t i = 0; i < input_names.size(); ++i) {
        vector<string> input_dims_str = caffe2::split(',', input_dims_list[i]);
        vector<int> input_dims;
        for (const string& s : input_dims_str) {
          input_dims.push_back(std::stoi(s));
        }
        caffe2::Blob* blob = workspace->GetBlob(input_names[i]);
        if (blob == nullptr) {
          blob = workspace->CreateBlob(input_names[i]);
        }
        if (run_on_gpu) {
          LOG(INFO) << "Running on GPU.";
#ifdef __CUDA_ARCH__
          caffe2::TensorCUDA* tensor = blob->GetMutable<caffe2::TensorCUDA>();
          TORCH_CHECK_NOTNULL(tensor);
          tensor->Resize(input_dims);
          if (input_type_list[i] == "uint8_t") {
            tensor->mutable_data<uint8_t>();
          } else if (input_type_list[i] == "float") {
            tensor->mutable_data<float>();
          } else {
            CAFFE_THROW("Unsupported input type: ", input_type_list[i]);
          }
#else
          CAFFE_THROW("Not support GPU on mobile.");
#endif
        } else {
          if (input_type_list[i] == "uint8_t") {
            caffe2::int8::Int8TensorCPU* tensor =
                blob->GetMutable<caffe2::int8::Int8TensorCPU>();
            TORCH_CHECK_NOTNULL(tensor);
            tensor->t.Resize(input_dims);
            tensor->t.mutable_data<uint8_t>();
          } else if (input_type_list[i] == "float") {
            caffe2::TensorCPU* tensor = BlobGetMutableTensor(blob, caffe2::CPU);
            TORCH_CHECK_NOTNULL(tensor);
            tensor->Resize(input_dims);
            tensor->mutable_data<float>();
          } else if (input_type_list[i] == "int") {
            caffe2::TensorCPU* tensor = BlobGetMutableTensor(blob, caffe2::CPU);
            TORCH_CHECK_NOTNULL(tensor);
            tensor->Resize(input_dims);
            tensor->mutable_data<int>();
          } else {
            CAFFE_THROW("Unsupported input type: ", input_type_list[i]);
          }
        }
      }
    } else {
      CAFFE_THROW(
          "You requested input tensors, but neither input_file nor "
          "input_dims is set.");
    }
  }
  return blob_num;
}

void fillInputBlob(
    shared_ptr<caffe2::Workspace> workspace,
    map<string, caffe2::TensorProtos>& tensor_protos_map,
    int iteration) {
  if (tensor_protos_map.empty()) {
    return;
  }
  static caffe2::TensorDeserializer deserializer;
  for (auto& tensor_kv : tensor_protos_map) {
    caffe2::Blob* blob = workspace->GetBlob(tensor_kv.first);
    if (blob == nullptr) {
      blob = workspace->CreateBlob(tensor_kv.first);
    }
    // todo: support gpu and make this function a template
    int protos_size = tensor_kv.second.protos_size();
    if (protos_size == 1 && iteration > 0) {
      // Do not override the input data if there is only one input data,
      // since it will clear all caches. Rely on wipe_cache to
      // clear caches
      continue;
    }
    caffe2::TensorProto* tensor_proto =
        tensor_kv.second.mutable_protos(iteration % protos_size);
    BlobSetTensor(blob, deserializer.Deserialize(*tensor_proto));
    // todo: for other types
  }
}

void runNetwork(
    shared_ptr<caffe2::Workspace> workspace,
    caffe2::NetBase* net,
    map<string, caffe2::TensorProtos>& tensor_protos_map,
    const bool wipe_cache,
    const bool run_individual,
    const bool run_on_gpu,
    const bool text_output,
    const int warmup,
    const int iter,
    const int num_blobs,
    const int sleep_before_run,
    const int sleep_between_iteration,
    const int sleep_between_net_and_operator,
    const std::string& output,
    const std::string& output_folder) {

  LOG(INFO) << "Starting benchmark.";
  caffe2::ObserverConfig::initSampleRate(1, 1, 1, run_individual, warmup);
  LOG(INFO) << "Running warmup runs.";
  for (int i = 0; i < warmup; ++i) {
    fillInputBlob(workspace, tensor_protos_map, i);
    CAFFE_ENFORCE(net->Run(), "Warmup run ", i, " has failed.");
  }

  if (wipe_cache) {
    caffe2::wipe_cache();
  }
  if (sleep_before_run > 0) {
    std::this_thread::sleep_for(std::chrono::seconds(sleep_before_run));
  }
  LOG(INFO) << "Main runs.";
  CAFFE_ENFORCE(
      iter >= 0,
      "Number of main runs should be non negative, provided ",
      iter,
      ".");
  LOG(INFO) << "net runs.";
  long long duration_sum = 0;
  for (int i = 0; i < iter; ++i) {
    caffe2::ObserverConfig::initSampleRate(1, 1, 1, 0, warmup);
    fillInputBlob(workspace, tensor_protos_map, i);
    if (wipe_cache) {
      caffe2::wipe_cache();
    }
    auto start = std::chrono::high_resolution_clock::now();
    CAFFE_ENFORCE(net->Run(), "Main run ", i, " has failed.");
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    duration_sum += duration.count();
    // Write the output for the first num_blobs times
    writeOutput(
        workspace,
        run_on_gpu,
        output,
        output_folder,
        text_output,
        i,
        num_blobs);
    if (wipe_cache) {
      caffe2::wipe_cache();
    }
    if (sleep_between_iteration > 0) {
      std::this_thread::sleep_for(
          std::chrono::seconds(sleep_between_iteration));
    }
  }
  std::cout << "Average Duration: " << (duration_sum/iter) << " us" << std::endl;
  if (run_individual) {
    LOG(INFO) << "operator runs.";
    if (sleep_between_net_and_operator > 0) {
      std::this_thread::sleep_for(
          std::chrono::seconds(sleep_between_net_and_operator));
    }
    for (int i = 0; i < iter; ++i) {
      caffe2::ObserverConfig::initSampleRate(1, 1, 1, 1, warmup);
      fillInputBlob(workspace, tensor_protos_map, i);
      CAFFE_ENFORCE(net->Run(), "Main run ", i, " with operator has failed.");
      if (wipe_cache) {
        caffe2::wipe_cache();
      }
      if (sleep_between_iteration > 0) {
        std::this_thread::sleep_for(
            std::chrono::seconds(sleep_between_iteration));
      }
    }
  }
}

void writeOutput(
    shared_ptr<caffe2::Workspace> workspace,
    const bool run_on_gpu,
    const string& output,
    const string& output_folder,
    const bool text_output,
    const int index,
    const int num_blobs) {
  if (output.size() == 0) {
    return;
  }
  string output_prefix = output_folder.size() ? output_folder + "/" : "";
  vector<string> output_names = caffe2::split(',', output);
  if (output == "*") {
    output_names = workspace->Blobs();
  }
  for (const string& name : output_names) {
    CAFFE_ENFORCE(
        workspace->HasBlob(name),
        "You requested a non-existing blob: ",
        name);
    if (text_output) {
      if (run_on_gpu) {
#ifdef __CUDA_ARCH__
        writeTextOutput<caffe2::CUDAContext, caffe2::TensorCUDA>(
            workspace->GetBlob(name)->GetMutable<caffe2::TensorCUDA>(),
            output_prefix,
            name,
            index,
            num_blobs);
#else
        CAFFE_THROW("Not support GPU.");
#endif
      } else {
        writeTextOutput<caffe2::CPUContext, caffe2::TensorCPU>(
            BlobGetMutableTensor(workspace->GetBlob(name), caffe2::CPU),
            output_prefix,
            name,
            index,
            num_blobs);
      }
    } else {
      // Do not support multiple entries per blob.
      CAFFE_ENFORCE(
          index == 0,
          "Binary file only support one output.");
      string serialized = SerializeBlob(*workspace->GetBlob(name), name);
      string output_filename = output_prefix + name;
      caffe2::WriteStringToFile(serialized, output_filename.c_str());
    }
  }
}

void logBenchmarkResult(
    const std::string& type,
    const std::string& metric,
    const std::string& unit,
    const int value) {
  LOG(INFO) << caffe2::NetObserverReporterPrint::IDENTIFIER << "{"
            << "\"type\": \"" << type << "\", "
            << "\"metric\": \"" << metric << "\", "
            << "\"unit\": \"" << unit << "\", "
            << "\"value\": " << c10::to_string(value) << "}\n";
}

long getVirtualMemoryIfOptionEnabled(bool FLAGS_measure_memory) {
  if (FLAGS_measure_memory) {
#if defined(TARGET_OS_IPHONE) || \
defined(TARGET_OS_MAC) || \
defined(TARGET_IPHONE_SIMULATOR)
    malloc_statistics_t stats = {0};
    malloc_zone_statistics(nullptr, &stats);
    return stats.size_allocated;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(
        GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    return pmc.PrivateUsage;
#else
    struct mallinfo info = mallinfo();
    return info.uordblks;
#endif
  }

  return 0;
}

int benchmark(
    int argc,
    char* argv[],
    const string& FLAGS_backend,
    const string& FLAGS_init_net,
    const string& FLAGS_input,
    const string& FLAGS_input_dims,
    const string& FLAGS_input_file,
    const string& FLAGS_input_type,
    int FLAGS_iter,
    bool FLAGS_measure_memory,
    const string& FLAGS_net,
    const string& FLAGS_output,
    const string& FLAGS_output_folder,
    bool FLAGS_run_individual,
    int FLAGS_sleep_before_run,
    int FLAGS_sleep_between_iteration,
    int FLAGS_sleep_between_net_and_operator,
    bool FLAGS_text_output,
    int FLAGS_warmup,
    bool FLAGS_wipe_cache) {
  // Check arguments to be correct
  {
    // Need to check whether file exists, as the file reader does not assert if
    // file does not exist
    std::ifstream net_file(FLAGS_net);
    CAFFE_ENFORCE(net_file.good());
    net_file.close();

    std::ifstream init_net_file(FLAGS_init_net);
    CAFFE_ENFORCE(init_net_file.good());
    init_net_file.close();

    if (FLAGS_input_file.size() > 0) {
      vector<string> input_files = caffe2::split(',', FLAGS_input_file);
      for (auto input_file : input_files) {
        std::ifstream ifile(input_file);
        CAFFE_ENFORCE(ifile.good());
        ifile.close();
      }
    }
  }

  observerConfig();
  caffe2::ShowLogInfoToStderr();

  auto workspace = std::make_shared<caffe2::Workspace>(new caffe2::Workspace());
  bool run_on_gpu = backendCudaSet(FLAGS_backend);
  // Run initialization network, measure resources used.
  long init_vmem = getVirtualMemoryIfOptionEnabled(FLAGS_measure_memory);
  caffe2::NetDef init_net_def;
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &init_net_def));
  setOperatorEngine(&init_net_def, FLAGS_backend);
  CAFFE_ENFORCE(workspace->RunNetOnce(init_net_def));
  init_vmem = getVirtualMemoryIfOptionEnabled(FLAGS_measure_memory) - init_vmem;

  map<string, caffe2::TensorProtos> tensor_protos_map;
  int num_blobs = loadInput(
      workspace,
      run_on_gpu,
      tensor_protos_map,
      FLAGS_input,
      FLAGS_input_file,
      FLAGS_input_dims,
      FLAGS_input_type);

  // Run main network.
  long predict_vmem = getVirtualMemoryIfOptionEnabled(FLAGS_measure_memory);
  caffe2::NetDef net_def;
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_net, &net_def));
  setOperatorEngine(&net_def, FLAGS_backend);
  if (!net_def.has_name()) {
    net_def.set_name("benchmark");
  }
  caffe2::NetBase* net = workspace->CreateNet(net_def);
  TORCH_CHECK_NOTNULL(net);
  runNetwork(
      workspace,
      net,
      tensor_protos_map,
      FLAGS_wipe_cache,
      FLAGS_run_individual,
      run_on_gpu,
      FLAGS_text_output,
      FLAGS_warmup,
      FLAGS_iter,
      num_blobs,
      FLAGS_sleep_before_run,
      FLAGS_sleep_between_iteration,
      FLAGS_sleep_between_net_and_operator,
      FLAGS_output,
      FLAGS_output_folder);
  predict_vmem = getVirtualMemoryIfOptionEnabled(
      FLAGS_measure_memory) - predict_vmem;
  if (FLAGS_measure_memory) {
    logBenchmarkResult(
        "NET_", "memory", "kB", (init_vmem + predict_vmem) / 1024);
  }

  return 0;
}
