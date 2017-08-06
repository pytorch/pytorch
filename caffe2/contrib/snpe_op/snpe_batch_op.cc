#if !(defined(__arm__) || defined(__aarch64__)) || !defined(__ANDROID__)
  #error "SNPE is available only for ARM/ARM64 Android and should not be used on other platforms"
#endif

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "snpe_ffi.h"
#include <dlfcn.h>

namespace caffe2 {

static constexpr const char* kExternalOutput = "__external_outputs__";

template <typename T>
using deleted_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

class SNPEBatchOp final : public Operator<CPUContext> {
 public:
  SNPEBatchOp(const OperatorDef& operator_def, Workspace* ws) : Operator<CPUContext>(operator_def, ws),
        model_buffer_(OperatorBase::GetSingleArgument<string>("model_buffer", ""))
  {
    handle_ = deleted_unique_ptr<void>(dlopen("libsnpe_jni.so", RTLD_LAZY), [](void* handle) {
      if (handle) {
        dlclose(handle);
      }
    });
    if (!handle_.get()) {
      std::cerr << dlerror() << std::endl;
    }

    OPERATOR_NEEDS_FEATURE(handle_.get(), "Couldn't find libsnpe_jni.so");

#define X(n)                                    \
  dlerror();                                    \
  auto* n##_f = (typeof(&n))dlsym(handle_.get(), #n); \
  OPERATOR_NEEDS_FEATURE(n##_f, dlerror());

    {
      X(snpe_has_gpu);
      X(snpe_create);
      X(snpe_destroy);
      X(snpe_get_input_dims);
      X(snpe_run);
      X(snpe_copy_output_to);
    }

    X(snpe_has_gpu);
    OPERATOR_NEEDS_FEATURE(snpe_has_gpu_f(), "No GPU found, cannot use SNPE.");

    X(snpe_create)
#undef X

// Redefine to use CAFFE_ENFORCE instead of OPERATOR_NEEDS_FEATURE.

#define X(n)                                              \
      dlerror();                                          \
      auto* n##_f = (typeof(&n))dlsym(handle_.get(), #n); \
      CAFFE_ENFORCE(n##_f, dlerror());

    const auto external_outputs_s = this->GetRepeatedArgument<std::string>(kExternalOutput);
    external_outputs.clear();
    for (auto s : external_outputs_s) {
      external_outputs.push_back(s.c_str());
    }

    ctx_ = deleted_unique_ptr<void>(snpe_create_f(reinterpret_cast<const unsigned char *>(model_buffer_.data()),
          model_buffer_.length(), external_outputs), [this](void* ctx) {
      if (ctx) {
        X(snpe_destroy);
        snpe_destroy_f(ctx);
      }
    });
  }

  bool RunOnDevice() override {
    X(snpe_get_input_dims);
    size_t const* dims;
    size_t dimSize;
    snpe_get_input_dims_f(ctx_.get(), &dims, &dimSize);
    CAFFE_ENFORCE_EQ(Input(0).ndim(), dimSize + 1);
    size_t N = Input(0).dim(0);
    CAFFE_ENFORCE(N >= 1);
    for (auto i = 0; i < dimSize; ++i) {
      CAFFE_ENFORCE_EQ(Input(0).dim(i + 1), dims[i]);
    }

    X(snpe_run);
    X(snpe_copy_output_to);

    // first batch
    auto input_single_batch = Input(0).size() / N;
    std::vector<size_t> output_single_batch(external_outputs.size(), 1);

    std::vector<std::vector<size_t>> outputsDims;
    snpe_run_f(ctx_.get(), Input(0).data<float>(),
      input_single_batch, external_outputs, &outputsDims);

    for (auto i = 0; i < outputsDims.size(); i++) {
      std::vector<size_t> batchOutputDims = outputsDims[i];
      batchOutputDims.insert(batchOutputDims.begin(), N);
      for (int dim = 1; dim < batchOutputDims.size(); dim++) {
        output_single_batch[i] *= batchOutputDims[dim];
      }
      Output(i)->Resize(batchOutputDims);
      snpe_copy_output_to_f(ctx_.get(),
        Output(i)->mutable_data<float>(),
        external_outputs[i]);

      CAFFE_ENFORCE(Output(i)->data<float>(), "nullptr where output should be!\n");
    };

    // remaining batches
    for (auto n = 1; n < N; n++) {
      snpe_run_f(ctx_.get(), Input(0).data<float>() + n * input_single_batch,
        input_single_batch, external_outputs, &outputsDims);
      for (auto i = 0; i < outputsDims.size(); i++) {
        snpe_copy_output_to_f(ctx_.get(),
          Output(i)->mutable_data<float>() + n * output_single_batch[i],
          external_outputs[i]);
      };
    }

    return true;
  }

 private:
  string model_buffer_;
  deleted_unique_ptr<void> handle_;
  // needs to be destroyed *before* handle_
  deleted_unique_ptr<void> ctx_;
  std::vector<const char*> external_outputs;
};

REGISTER_CPU_OPERATOR(SNPEBatch, SNPEBatchOp);
}

#undef X
