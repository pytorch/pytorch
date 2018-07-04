#include "caffe2/mkl/mkl_utils.h"

#include "caffe2/core/init.h"
#include "caffe2/core/tensor.h"

#ifdef CAFFE2_HAS_MKL_DNN

CAFFE2_DEFINE_bool(
    caffe2_mkl_implicit_layout_change, false,
    "Controls the behavior when we call View() on an MKLMemory: if it is set "
    "true, then the View() function will actually change the underlying "
    "storage. If it is set false, an implicit copy is triggered but the "
    "original storage is not affected."
    );

namespace caffe2 {

CAFFE_KNOWN_TYPE(mkl::MKLMemory<float>);
CAFFE_KNOWN_TYPE(mkl::MKLMemory<double>);

template <typename T>
static vector<TIndex> GetMKLTensorInfo(
    const void* c,
    bool* shares_data,
    size_t* capacity,
    DeviceOption* device) {
  const mkl::MKLMemory<T>* tc = static_cast<const mkl::MKLMemory<T>*>(c);
  // it's too hard to get sharing info from mkl::MKLMemory
  *shares_data = false;
  *capacity = tc->size() * sizeof(T);
  device->set_device_type(MKLDNN);
  device->set_cuda_gpu_id(0);
  return tc->dims();
}

template <typename T>
static TypeMeta GetMKLTensorType(const void*) {
  return TypeMeta::Make<T>();
}

template <typename T>
static void RegisterForType() {
  RegisterTypeCallFunction(
      TypeMeta::Id<mkl::MKLMemory<T>>(), GetMKLTensorType<T>);
  RegisterTensorInfoFunction(
      TypeMeta::Id<mkl::MKLMemory<T>>(), GetMKLTensorInfo<T>);
}

static bool Caffe2InitializeMKL(int*, char***) {
  RegisterForType<float>();
  RegisterForType<double>();
  return true;
}

REGISTER_CAFFE2_INIT_FUNCTION(
    InitMKLDNNContext,
    &Caffe2InitializeMKL,
    "Register wrappers for MKLContext");

} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
