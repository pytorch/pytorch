#include <c10/core/CopyBytes.h>
#include <c10/util/Logging.h>

namespace c10 {

// First dimension of the array is `bool async`: 0 is sync,
// 1 is async (non-blocking)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
static CopyBytesFunction g_copy_bytes[2][COMPILE_TIME_MAX_DEVICE_TYPES]
                                     [COMPILE_TIME_MAX_DEVICE_TYPES];

_CopyBytesFunctionRegisterer::_CopyBytesFunctionRegisterer(
    DeviceType fromType,
    DeviceType toType,
    CopyBytesFunction func_sync,
    CopyBytesFunction func_async) {
  auto from = static_cast<int>(fromType);
  auto to = static_cast<int>(toType);
  if (!func_async) {
    // default to the sync function
    func_async = func_sync;
  }
  CHECK(
      g_copy_bytes[0][from][to] == nullptr &&
      g_copy_bytes[1][from][to] == nullptr)
      << "Duplicate registration for device type pair "
      << c10::DeviceTypeName(fromType) << ", " << c10::DeviceTypeName(toType);
  g_copy_bytes[0][from][to] = func_sync;
  g_copy_bytes[1][from][to] = func_async;
}

void CopyBytes(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device,
    bool async) {
  auto ptr = g_copy_bytes[async ? 1 : 0][static_cast<int>(src_device.type())]
                         [static_cast<int>(dst_device.type())];
  CAFFE_ENFORCE(
      ptr,
      "No function found for copying from ",
      c10::DeviceTypeName(src_device.type()),
      " to ",
      c10::DeviceTypeName(dst_device.type()));
  ptr(nbytes, src, src_device, dst, dst_device);
}

} // namespace c10
