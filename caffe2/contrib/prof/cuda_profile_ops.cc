#include "caffe2/core/common_gpu.h"
#include "caffe2/core/operator.h"

#include <stdlib.h>
#include <string.h>

#include <cuda_profiler_api.h>

namespace caffe2 {

static std::vector<std::string> kCudaProfileConfiguration = {
    "gpustarttimestamp",
    "gpuendtimestamp",
    "gridsize3d",
    "threadblocksize",
    "dynsmemperblock",
    "stasmemperblock",
    "regperthread",
    "memtransfersize",
    "memtransferdir",
    "memtransferhostmemtype",
    "streamid",
    "cacheconfigrequested",
    "cacheconfigexecuted",
    "countermodeaggregate",
    "enableonstart 0",
    "active_warps",
    "active_cycles",
};

class CudaProfileInitializeOp : public OperatorBase {
 public:
  CudaProfileInitializeOp(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws),
        output_(GetSingleArgument<std::string>("output", "/tmp/output")) {
    std::array<char, 128> buf;
    std::string tmpl = "/tmp/cuda_profile_config.XXXXXX";
    CAFFE_ENFORCE_LT(tmpl.size(), buf.size());
    memcpy(buf.data(), tmpl.data(), tmpl.size());
    auto result = mktemp(buf.data());
    CAFFE_ENFORCE_NE(strlen(result), 0, "mktemp: ", strerror(errno));
    config_ = result;

    // Write configuration to temporary file
    {
      std::ofstream ofs(config_, std::ios::out | std::ios::trunc);
      CAFFE_ENFORCE(ofs.is_open(), "ofstream: ", ofs.rdstate());
      for (const auto& line : kCudaProfileConfiguration) {
        ofs << line << std::endl;
      }
    }
  }

  ~CudaProfileInitializeOp() override {
    unlink(config_.c_str());
  }

  bool Run(int /* unused */ /*stream_id*/ = 0) override {
    // If this fails, check the contents of "output" for hints.
    CUDA_CHECK(
        cudaProfilerInitialize(config_.c_str(), output_.c_str(), cudaCSV));
    return true;
  }

 protected:
  std::string config_;
  std::string output_;
};

class CudaProfileStartOp : public OperatorBase {
 public:
  CudaProfileStartOp(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws) {}

  bool Run(int /* unused */ /*stream_id*/ = 0) override {
    CUDA_ENFORCE(cudaProfilerStart());
    return true;
  }
};

class CudaProfileStopOp : public OperatorBase {
 public:
  CudaProfileStopOp(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws) {}

  bool Run(int /* unused */ /*stream_id*/ = 0) override {
    CUDA_ENFORCE(cudaProfilerStop());
    return true;
  }
};

OPERATOR_SCHEMA(CudaProfileInitialize);
OPERATOR_SCHEMA(CudaProfileStart);
OPERATOR_SCHEMA(CudaProfileStop);

REGISTER_CPU_OPERATOR(CudaProfileInitialize, CudaProfileInitializeOp);
REGISTER_CPU_OPERATOR(CudaProfileStart, CudaProfileStartOp);
REGISTER_CPU_OPERATOR(CudaProfileStop, CudaProfileStopOp);

REGISTER_CUDA_OPERATOR(CudaProfileInitialize, CudaProfileInitializeOp);
REGISTER_CUDA_OPERATOR(CudaProfileStart, CudaProfileStartOp);
REGISTER_CUDA_OPERATOR(CudaProfileStop, CudaProfileStopOp);

} // namespace caffe2
