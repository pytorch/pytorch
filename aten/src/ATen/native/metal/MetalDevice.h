#ifndef PYTORCH_MOBILE_METAL_DEVICE_H_
#define PYTORCH_MOBILE_METAL_DEVICE_H_

#import <Metal/Metal.h>

#include <string>

namespace at {
namespace native {
namespace metal {

struct MetalDeviceInfo {
  std::string name;
  MTLLanguageVersion languageVersion;
};

static inline MetalDeviceInfo createDeviceInfo(id<MTLDevice> device) {
  MetalDeviceInfo device_info;
  device_info.name = device.name.UTF8String;
  if (@available(macOS 11.0, iOS 14.0, *)) {
    device_info.languageVersion = MTLLanguageVersion2_3;
  } else if (@available(macOS 10.15, iOS 13.0, *)) {
    device_info.languageVersion = MTLLanguageVersion2_2;
  } else if (@available(macOS 10.14, iOS 12.0, *)) {
    device_info.languageVersion = MTLLanguageVersion2_1;
  } else if (@available(macOS 10.13, iOS 11.0, *)) {
    device_info.languageVersion = MTLLanguageVersion2_0;
  } else if (@available(macOS 10.12, iOS 10.0, *)) {
    device_info.languageVersion = MTLLanguageVersion1_2;
  } else if (@available(macOS 10.11, iOS 9.0, *)) {
    device_info.languageVersion = MTLLanguageVersion1_1;
  }
#if (                                                    \
    defined(__IPHONE_9_0) &&                             \
    __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_9_0) || \
    (defined(__MAC_10_11) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_11)
#else
#error "Metal is not available on the current platform."
#endif
  return device_info;
}

}
}
}

#endif
