#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "snpe_ffi.h"
#include <dlfcn.h>

namespace caffe2 {

template <typename T>
using deleted_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

class SNPEOp final : public Operator<CPUContext> {
 public:
  SNPEOp(const OperatorDef& def, Workspace* ws) : Operator<CPUContext>(def, ws),
        model_buffer_(OperatorBase::GetSingleArgument<string>("model_buffer", "")),
        input_name_(OperatorBase::GetSingleArgument<string>("input_name", "data"))
  {
    CAFFE_ENFORCE(gSNPELocation() != "", "SNPE library \"", gSNPELocation(), "\" does not exist.");
    std::ostringstream snpe_ffi;
    snpe_ffi << gSNPELocation() << "/" << snpe_ffi_so;
    handle_ = deleted_unique_ptr<void>(dlopen(snpe_ffi.str().c_str(), RTLD_LAZY), [](void* handle) {
      if (handle) {
        dlclose(handle);
      }
    });
    if (!handle_.get()) {
      std::cerr << dlerror() << std::endl;
    }

    OPERATOR_NEEDS_FEATURE(handle_.get(), "Couldn't find ", snpe_ffi.str());

#define X(n)                                    \
  dlerror();                                    \
  auto* n##_f = (decltype(&n))dlsym(handle_.get(), #n); \
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
      auto* n##_f = (decltype(&n))dlsym(handle_.get(), #n); \
      CAFFE_ENFORCE(n##_f, dlerror());

    CAFFE_ENFORCE(def.input_size(), "No inputs.");
    if (input_name_ == "") {
      input_name_ = def.input().Get(0);
		}
    ctx_ = deleted_unique_ptr<void>(snpe_create_f(reinterpret_cast<const unsigned char *>(model_buffer_.data()),
          model_buffer_.length(), input_name_.c_str()), [this](void* ctx) {
      if (ctx) {
        X(snpe_destroy);
        snpe_destroy_f(ctx);
      }
    });
  }

  bool RunOnDevice() override {
    CAFFE_ENFORCE(gSNPELocation() != "", "SNPE library was never loaded.");

    X(snpe_get_input_dims);
    size_t const* dims;
    size_t dimSize;
    snpe_get_input_dims_f(ctx_.get(), &dims, &dimSize);
    if (Input(0).ndim() != dimSize) {
      if (dimSize == 3 && dimSize == Input(0).ndim() - 1 && Input(0).dim32(0) == 1) {
        const int C = Input(0).dim32(1);
        const int H = Input(0).dim32(2);
        const int W = Input(0).dim32(3);
        if (dims[0] != C ||
            dims[1] != H ||
            dims[2] != W) {
          CAFFE_THROW("Input size must match what SNPE expects, which in this case is: ",
              dims[0], " ", dims[1], " ", dims[2]);
        }
      } else {
        CAFFE_THROW("SNPE input dimensions are not compatible.");
      }
    } else {
      for (auto i = 0; i < Input(0).ndim(); ++i) {
        CAFFE_ENFORCE_EQ(dims[i], Input(0).dim32(i), "SNPE input dimension is not compatible.");
      }
		}

    X(snpe_run);
    CAFFE_ENFORCE(ctx_.get(), "SNPE context doesn't exist.");
    snpe_run_f(ctx_.get(), Input(0).data<float>(), Input(0).size(), &dims, &dimSize);

    std::vector<int64_t> outputDims(dimSize + 1);
    outputDims[0] = 1;
    for (auto i = 0; i < dimSize; ++i) {
      outputDims[i+1] = dims[i];
    };

    Output(0)->Resize(outputDims);
    X(snpe_copy_output_to);
    snpe_copy_output_to_f(ctx_.get(), Output(0)->mutable_data<float>());

    CAFFE_ENFORCE(Output(0)->data<float>(), "nullptr where output should be!\n");
    return true;
  }

 private:
  string model_buffer_;
  string input_name_;
  deleted_unique_ptr<void> handle_;
  // needs to be destroyed *before* handle_
  deleted_unique_ptr<void> ctx_;
};

REGISTER_CPU_OPERATOR(SNPE, SNPEOp);
}

#undef X
