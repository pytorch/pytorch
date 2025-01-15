#import <ATen/native/metal/MetalDevice.h>
#import <ATen/native/metal/MetalShaders.h>
#import <ATen/native/metal/MetalContext.h>

#include <c10/util/Exception.h>

#include <mutex>
#include <unordered_map>

#if C10_IOS
#import <UIKit/UIKit.h>
#elif TARGET_OS_MAC
#import <Foundation/NSProcessInfo.h>
#endif

using namespace at::native::metal;
@implementation MetalContext {
  std::mutex _pipelineCacheMutex;
  MetalDeviceInfo _deviceInfo;
  std::unordered_map<std::string, id<MTLComputePipelineState>> _pipelineCache;
}

+ (instancetype)sharedInstance {
  static dispatch_once_t onceToken;
  static MetalContext* instance = nil;
  dispatch_once(&onceToken, ^{
    instance = [[MetalContext alloc] init];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    instance->_device = device;
    instance->_deviceInfo = createDeviceInfo(device);
    instance->_library = nil;
    instance->_commandQueue = [instance.device newCommandQueue];
  });
  return instance;
}

- (BOOL)available {
#if !defined(__APPLE__)
  return false;
#elif TARGET_OS_IPHONE
  if (!MPSSupportsMTLDevice(_device)) {
    return false;
  }
  if ([UIDevice currentDevice].systemVersion.floatValue < 11.0) {
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
C10_CLANG_DIAGNOSTIC_PUSH()
C10_CLANG_DIAGNOSTIC_IGNORE("-Wdeprecated-declarations")
  if (![_device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily1_v3]) {
    return false;
  }
C10_CLANG_DIAGNOSTIC_POP()
#else
  return false;
#endif
  NSError* error = [self _compileProgram];
  if (error) {
    std::string compilationError = error.localizedDescription.UTF8String;
    std::string deviceInfo = self.description.UTF8String;
    TORCH_CHECK(false, compilationError + "\n" + deviceInfo);
  }
  return _device && _library && _commandQueue;
}

- (id<MTLComputePipelineState>)pipelineState:(const std::string&)kernel {
  TORCH_CHECK(_library, "Failed to load Metal shaders");
  std::lock_guard<std::mutex> g(_pipelineCacheMutex);
  id<MTLComputePipelineState> state = _pipelineCache[kernel];
  if (state) {
    return state;
  }
  id<MTLFunction> func = [_library newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()] ?: @""];
  TORCH_CHECK(func, "Failed to load the Metal Shader function: ", kernel);
  NSError* errors = nil;
  state = [_device newComputePipelineStateWithFunction:func error:&errors];
  TORCH_CHECK(state, errors.localizedDescription.UTF8String);
  _pipelineCache[kernel] = state;
  return state;
}

- (id<MTLComputePipelineState>)specializedPipelineState:(const std::string&)kernel
                                              Constants:(NSArray<NSNumber*>*)
                                                            constants {
  TORCH_CHECK(_library, "Failed to load Metal shaders");
  std::string kernelStr = kernel;
  for (NSUInteger i = 0; i < constants.count; ++i) {
    kernelStr += "_" + std::string([constants[i] stringValue].UTF8String);
  }
  std::lock_guard<std::mutex> g(_pipelineCacheMutex);
  id<MTLComputePipelineState> state = _pipelineCache[kernelStr];
  if (state) {
    return state;
  }
  MTLFunctionConstantValues* constantValues = [MTLFunctionConstantValues new];
  NSUInteger ushortArgIndex = 0;
  NSUInteger floatArgIndex = 12;
  for (NSUInteger i = 0; i < constants.count; ++i) {
    NSNumber* constant = constants[i];
    const char* type = constant.objCType;
    if (strcmp(type, @encode(NSUInteger)) == 0 ||
        strcmp(type, @encode(NSInteger)) == 0) {
      TORCH_CHECK(ushortArgIndex <= 12);
      ushort value = ushort([constant unsignedIntegerValue]);
      [constantValues setConstantValue:&value
                                  type:MTLDataTypeUShort
                               atIndex:ushortArgIndex];
      ushortArgIndex++;
    }
    if (strcmp(type, @encode(float)) == 0 ||
        strcmp(type, @encode(double)) == 0) {
      TORCH_CHECK(floatArgIndex <= 14);
      float value = [constant floatValue];
      [constantValues setConstantValue:&value
                                  type:MTLDataTypeFloat
                               atIndex:floatArgIndex];
      floatArgIndex++;
    }
  }
  NSError* errors = nil;
  id<MTLFunction> func = [_library newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()] ?: @""
                                        constantValues:constantValues
                                                 error:&errors];
  TORCH_CHECK(func, errors.localizedDescription.UTF8String);
  state = [_device newComputePipelineStateWithFunction:func error:&errors];
  TORCH_CHECK(state, errors.localizedDescription.UTF8String);
  _pipelineCache[kernelStr] = state;
  return state;
}

- (id<MTLBuffer>)emptyMTLBuffer:(int64_t) size {
    TORCH_CHECK(_device);
    id<MTLBuffer> buffer = [_device newBufferWithLength:size
                      options:MTLResourceCPUCacheModeWriteCombined];
    return buffer;
}

- (NSString*)description {
  NSString* desc =
      [NSString stringWithFormat:@"DeviceName: %s, LanguageVersion: %lu",
                                 _deviceInfo.name.c_str(),
                                 (unsigned long)_deviceInfo.languageVersion];
  return desc;
}

- (NSError*)_compileProgram {
  __block NSError* compilationError = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    NSError* localError = nil;
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    [options setLanguageVersion:_deviceInfo.languageVersion];
    [options setFastMathEnabled:YES];
    _library = [_device
        newLibraryWithSource:[NSString stringWithUTF8String:PT_METAL_SHADERS] ?: @""
                     options:options
                       error:&localError];
    compilationError = localError;
  });
  return compilationError;
}



@end
