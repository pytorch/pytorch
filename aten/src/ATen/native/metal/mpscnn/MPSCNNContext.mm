#import <ATen/native/metal/MetalShaders.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>

#include <c10/util/Exception.h>

#include <mutex>

#if C10_IOS
#import <UIKit/UIKit.h>
#elif TARGET_OS_MAC
#import <Foundation/NSProcessInfo.h>
#endif

@implementation MPSCNNContext {
  std::mutex _pipelineCacheMutex;
  NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* _pipelineCache;
}

+ (instancetype)sharedInstance {
  static dispatch_once_t onceToken;
  static MPSCNNContext* instance = nil;
  dispatch_once(&onceToken, ^{
    instance = [[MPSCNNContext alloc] init];
    instance->_device = MTLCreateSystemDefaultDevice();
    instance->_library = [instance.device
        newLibraryWithSource:[NSString stringWithUTF8String:PT_METAL_SHADERS]
                     options:nil
                       error:nil];
    instance->_commandQueue = [instance.device newCommandQueue];
    instance->_pipelineCache =
        [NSMutableDictionary<NSString*, id<MTLComputePipelineState>> new];
  });
  return instance;
}

- (BOOL)available {
#if !defined(__APPLE__)
  return false;
#elif TARGET_IPHONE_SIMULATOR
  return false;
#elif TARGET_OS_IPHONE
  if (!MPSSupportsMTLDevice(_device)) {
    return false;
  }
  if ([UIDevice currentDevice].systemVersion.floatValue < 10.2) {
    return false;
  }
  if (![MTLCreateSystemDefaultDevice()
          supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily3_v2]) {
    return false;
  }
#elif TARGET_OS_MAC
  if (!MPSSupportsMTLDevice(_device)) {
    return false;
  }
  NSOperatingSystemVersion supportedVer = {10, 13, 0};
  if (![[NSProcessInfo processInfo]
          isOperatingSystemAtLeastVersion:supportedVer]) {
    return false;
  }
  if (![MTLCreateSystemDefaultDevice()
          supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily1_v3]) {
    return false;
  }
#else
  return false;
#endif

  return _device && _library && _commandQueue;
}

- (id<MTLComputePipelineState>)pipelineState:(NSString*)kernel {
  TORCH_CHECK(_library, "Failed to load Metal shaders");
  std::lock_guard<std::mutex> g(_pipelineCacheMutex);
  id<MTLComputePipelineState> state = _pipelineCache[kernel];
  if (state) {
    return state;
  }
  id<MTLFunction> func = [_library newFunctionWithName:kernel];
  TORCH_CHECK(func, "Failed to load the Metal Shader function: ", kernel);
  NSError* errors;
  state = [_device newComputePipelineStateWithFunction:func error:&errors];
  TORCH_CHECK(state, errors.localizedDescription.UTF8String);
  _pipelineCache[kernel] = state;
  return state;
}

- (id<MTLComputePipelineState>)specializedPipelineState:(NSString*)kernel
                                              Constants:(NSArray<NSNumber*>*)
                                                            constants {
  TORCH_CHECK(_library, "Failed to load Metal shaders");
  std::string kernelStr = std::string([kernel UTF8String]);
  for (auto i = 0; i < constants.count; ++i) {
    kernelStr += "_" + std::string([constants[i] stringValue].UTF8String);
  }
  std::lock_guard<std::mutex> g(_pipelineCacheMutex);
  id<MTLComputePipelineState> state = _pipelineCache[kernel];
  if (state) {
    return state;
  }
  MTLFunctionConstantValues* constantValues = [MTLFunctionConstantValues new];
  NSUInteger ushortArgIndex = 0;
  NSUInteger floatArgIndex = 10;
  for (auto i = 0; i < constants.count; ++i) {
    NSNumber* constant = constants[i];
    const char* type = constant.objCType;
    if (strcmp(type, @encode(NSUInteger)) == 0 ||
        strcmp(type, @encode(NSInteger)) == 0) {
      TORCH_CHECK(ushortArgIndex <= 10);
      ushort value = ushort([constant unsignedIntegerValue]);
      [constantValues setConstantValue:&value
                                  type:MTLDataTypeUShort
                               atIndex:ushortArgIndex];
      ushortArgIndex++;
    }
    if (strcmp(type, @encode(float)) == 0 ||
        strcmp(type, @encode(double)) == 0) {
      TORCH_CHECK(floatArgIndex <= 2);
      float value = [constant floatValue];
      [constantValues setConstantValue:&value
                                  type:MTLDataTypeFloat
                               atIndex:floatArgIndex];
      floatArgIndex++;
    }
  }
  NSError* errors;
  id<MTLFunction> func = [_library newFunctionWithName:kernel
                                        constantValues:constantValues
                                                 error:&errors];
  TORCH_CHECK(func, errors.localizedDescription.UTF8String);
  state = [_device newComputePipelineStateWithFunction:func error:&errors];
  TORCH_CHECK(state, errors.localizedDescription.UTF8String);
  kernel = [NSString stringWithCString:kernelStr.c_str()
                              encoding:NSUTF8StringEncoding];
  _pipelineCache[kernel] = state;
  return state;
}

@end
