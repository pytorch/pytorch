#include <torch/csrc/jit/backends/coreml/objc/PTMCoreMLExecutor.h>
#include <torch/script.h>

#import <CoreML/CoreML.h>

#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#endif

// Observer
#import <torch/csrc/jit/backends/coreml/observer/PTMCoreMLObserver.h>

#include <sys/utsname.h>
#include <fstream>
#include <iostream>

// This is a utility macro that can be used to throw an exception when a CoreML
// API function produces a NSError. The exception will contain a message with
// useful info extracted from the NSError.
#define COREML_THROW_IF_ERROR(error, preamble)                                   \
  do {                                                                           \
    if C10_LIKELY(error) {                                                       \
      throw c10::Error(                                                          \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},                 \
          c10::str(                                                              \
              preamble,                                                          \
              " Error details: ",                                                \
              " Localized_description: ", error.localizedDescription.UTF8String, \
              " Domain: ", error.domain.UTF8String,                              \
              " Code: ", error.code,                                             \
              " User Info: ", error.userInfo.description.UTF8String));           \
    }                                                                            \
  } while (false)

@implementation PTMCoreMLFeatureProvider {
  NSUInteger _coremlVersion;
  std::vector<PTMCoreMLFeatureSpecs> _specs;
}

@synthesize featureNames = _featureNames;

- (instancetype)initWithFeatureSpecs:
                    (const std::vector<PTMCoreMLFeatureSpecs>&)specs
                       CoreMLVersion:(NSUInteger)ver {
  self = [super init];
  if (self) {
    _coremlVersion = ver;
    _specs = specs;
    NSMutableArray* names = [NSMutableArray new];
    for (auto& spec : _specs) {
      [names addObject:spec.name];
    }
    _featureNames = [[NSSet alloc] initWithArray:names];
  }
  return self;
}

- (nullable MLFeatureValue*)featureValueForName:(NSString*)featureName {
  for (auto& spec : _specs) {
    if ([spec.name isEqualToString:featureName]) {
      NSMutableArray* shape = [NSMutableArray new];
      for (auto& dim : spec.tensor.sizes().vec()) {
        [shape addObject:@(dim)];
      }
      NSMutableArray* strides = [NSMutableArray new];
      for (auto& step : spec.tensor.strides().vec()) {
        [strides addObject:@(step)];
      }
      NSError* error = nil;
      TORCH_CHECK(spec.tensor.dtype() == c10::kFloat);
      MLMultiArray* mlArray = [[MLMultiArray alloc]
          initWithDataPointer:spec.tensor.data_ptr<float>()
                        shape:shape
                     dataType:MLMultiArrayDataTypeFloat32
                      strides:strides
                  deallocator:(^(void* bytes){
                              })error:&error];
      return [MLFeatureValue featureValueWithMultiArray:mlArray];
    }
  }
  return nil;
}

@end

static NSString* gModelCacheDirectory = @"";

@implementation PTMCoreMLExecutor {
  MLModel* _mlModel;
  NSURL* _modelPath;
  NSURL* _compiledModelPath;

  int32_t _model_load_id;
  int32_t _inferences;

  int32_t _sample_thresh;
  int32_t _sample_every;

  size_t _init_mem_limit;
}

+ (void)setModelCacheDirectory:(NSString*)dir {
  gModelCacheDirectory = dir;
}

+ (NSString*)modelCacheDirectory {
  if (gModelCacheDirectory.length == 0 ||
      ![[NSFileManager defaultManager]
          isWritableFileAtPath:gModelCacheDirectory]) {
    // set the default directory to tmp
    gModelCacheDirectory = NSTemporaryDirectory();
  }
  return gModelCacheDirectory;
}

+ (BOOL)isAvailable {
#if !defined(__APPLE__)
  return false;
#elif TARGET_OS_IPHONE
  if ([UIDevice currentDevice].systemVersion.floatValue > 14.0) {
    return true;
  }
#elif TARGET_OS_MAC
  NSOperatingSystemVersion supportedVer = {10, 13, 0};
  if ([[NSProcessInfo processInfo]
          isOperatingSystemAtLeastVersion:supportedVer]) {
    return true;
  }
#endif
  return false;
}

- (BOOL)compileMLModel:(const std::string&)modelSpecs
            identifier:(const std::string&)identifier
    API_AVAILABLE(ios(11.0), macos(10.13)) {
  NSString* mlModelName = [NSString stringWithCString:identifier.c_str()
                                             encoding:NSUTF8StringEncoding];
  _modelPath = [self _cacheFilePath:mlModelName];
  [self _saveModel:modelSpecs];
  NSError* error = nil;
  _compiledModelPath = [self _compiledModelFilePath:_modelPath.path];

  // Get observer and create an instance key
  PTMCoreMLObserver* observer = coreMLObserverConfig().getCoreMLObserver();
  int32_t instance_key = std::rand();
  _model_load_id = std::rand();
  _inferences = 0;

  _init_mem_limit = 0;

  _sample_thresh =
      static_cast<int32_t>(1.0 / 1000.0 * static_cast<double>(RAND_MAX));
  _sample_every = 500;

  if (observer) {
    _init_mem_limit = observer->getRemainingMemory();
    observer->onEnterCompileModel(instance_key, _model_load_id);
  }

  // Compile the model when OS version changes
  if ([self _shouldRecompileModel]) {
    if (@available(iOS 11.0, macOS 10.13, *)) {
      NSURL* temporaryFileURL = [MLModel compileModelAtURL:_modelPath
                                                     error:&error];
      if (!error) {
        // move the model to the cache directory
        NSFileManager* fileManager = [NSFileManager defaultManager];
        if (![temporaryFileURL isEqual:_compiledModelPath]) {
          if ([fileManager fileExistsAtPath:_compiledModelPath.path]) {
            [fileManager removeItemAtURL:_compiledModelPath error:&error];
          }
          [fileManager moveItemAtURL:temporaryFileURL
                               toURL:_compiledModelPath
                               error:&error];
        }
      }
    } else {
      // Always log on failure
      if (observer) {
        observer->onExitCompileModel(instance_key, false, true);
      }
      TORCH_CHECK(false, "CoreML is not available on your deivce");
    }
  }

  if (error) {
    // Always log on failure
    if (observer) {
      observer->onExitCompileModel(instance_key, false, true);
    }

    // remove cached models if compalition failed.
    [self cleanup];

    COREML_THROW_IF_ERROR(error, "Error compiling the MLModel file!");
    return NO;
  }
  if (@available(iOS 12.0, macOS 10.14, *)) {
    MLModelConfiguration* config = [MLModelConfiguration alloc];
    MLComputeUnits backend = MLComputeUnitsCPUOnly;
    if ([self.backend isEqualToString:@"cpuandgpu"]) {
      backend = MLComputeUnitsCPUAndGPU;
    } else if ([self.backend isEqualToString:@"all"]) {
      backend = MLComputeUnitsAll;
    }
    config.computeUnits = backend;
    config.allowLowPrecisionAccumulationOnGPU = self.allowLowPrecision;
    _mlModel = [MLModel modelWithContentsOfURL:_compiledModelPath
                                 configuration:config
                                         error:&error];
  } else {
    _mlModel = [MLModel modelWithContentsOfURL:_compiledModelPath error:&error];
  }
  if (error || !_mlModel) {
    // Always log on failure
    if (observer) {
      observer->onExitCompileModel(instance_key, false, true);
    }

    COREML_THROW_IF_ERROR(error, "Error loading the MLModel file!");
  }

  if (observer) {
    bool should_log = _model_load_id < _sample_thresh;
    observer->onExitCompileModel(instance_key, true, should_log);
  }

  return YES;
}

- (id<MLFeatureProvider>)forwardWithInputs:
    (const std::vector<PTMCoreMLFeatureSpecs>&)inputs {
  @autoreleasepool {
    // Get observer and create an instance key
    PTMCoreMLObserver* observer = coreMLObserverConfig().getCoreMLObserver();
    int32_t instance_key = std::rand();

    if (observer) {
      observer->onEnterExecuteModel(
          instance_key, _model_load_id, _init_mem_limit, _inferences);
    }

    NSError* error = nil;
    PTMCoreMLFeatureProvider* inputFeature = [[PTMCoreMLFeatureProvider alloc]
        initWithFeatureSpecs:inputs
               CoreMLVersion:self.coreMLVersion];
    if (inputFeature == nil) {
      return nil;
    }
    if (@available(iOS 11.0, macOS 10.13, *)) {
      MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
      id<MLFeatureProvider> outputFeature =
          [_mlModel predictionFromFeatures:inputFeature
                                   options:options
                                     error:&error];

      COREML_THROW_IF_ERROR(error, "Error running CoreML inference!");

      ++_inferences;
      if (observer) {
        // Check if this inference session is being logged.
        // If so, only log every N inferences
        bool should_log = _model_load_id < _sample_thresh && _inferences > 1;
        if (should_log) {
          should_log = _inferences % _sample_every == 0;
        }
        observer->onExitExecuteModel(
            instance_key, _inferences, true, should_log);
      }

      return outputFeature;
    } else {
      // Always log on failure
      if (observer) {
        observer->onExitExecuteModel(instance_key, _inferences, true, true);
      }

      TORCH_CHECK(false, "Core ML is not available on your device");
      return nil;
    }
  }
}

- (BOOL)cleanup {
  NSFileManager* fileManager = [NSFileManager defaultManager];
  NSError* error = nil;
  NSString* modelPath = _modelPath.path;
  NSString* compiledModelPath = _compiledModelPath.path;
  if ([fileManager fileExistsAtPath:modelPath]) {
    [fileManager removeItemAtPath:modelPath error:&error];
  }
  if ([fileManager fileExistsAtPath:compiledModelPath]) {
    [fileManager removeItemAtPath:compiledModelPath error:&error];
  }
  return !error;
}

- (void)_saveModel:(const std::string&)spec {
  NSString* modelPath = _modelPath.path;
  if (![[NSFileManager defaultManager] fileExistsAtPath:modelPath]) {
    // Note that the serialized protobuf binary contains bytes, not text;
    // see
    // https://developers.google.com/protocol-buffers/docs/pythontutorial#parsing-and-serialization
    NSData* data = [NSData dataWithBytes:spec.c_str() length:spec.length()];
    BOOL ret = [data writeToFile:modelPath atomically:YES];
    TORCH_CHECK(ret, "Error saving the MLModel", modelPath.UTF8String);
  }
}

- (BOOL)_shouldRecompileModel {
#if TARGET_OS_IPHONE
  NSError* error = nil;
  NSString* currentOSVer = [UIDevice currentDevice].systemVersion;
  NSString* versionPath = [self _cacheFilePath:@"version"].path;
  BOOL shouldRecompileModel = YES;
  NSFileManager* fileManager = [NSFileManager defaultManager];
  if ([fileManager fileExistsAtPath:versionPath]) {
    NSString* cachedOSVer =
        [NSString stringWithContentsOfFile:versionPath
                                  encoding:NSUTF8StringEncoding
                                     error:&error];
    if ([cachedOSVer isEqualToString:currentOSVer]) {
      if ([fileManager fileExistsAtPath:_compiledModelPath.path]) {
        shouldRecompileModel = NO;
      }
    }
  }
  [currentOSVer writeToFile:versionPath atomically:YES];
  return shouldRecompileModel;
#else
  return YES;
#endif
}

- (NSURL*)_cacheFilePath:(NSString*)fileName {
  NSString* filePath = [[[self class] modelCacheDirectory]
      stringByAppendingPathComponent:fileName];
  return [NSURL fileURLWithPath:filePath];
}

- (NSURL*)_compiledModelFilePath:(NSString*)modelPath {
  NSString* filePath = [modelPath stringByAppendingString:@".mlmodelc"];
  return [NSURL fileURLWithPath:filePath];
}

@end
