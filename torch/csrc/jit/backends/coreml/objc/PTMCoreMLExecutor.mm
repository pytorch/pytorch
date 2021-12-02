#include <torch/csrc/jit/backends/coreml/objc/PTMCoreMLExecutor.h>
#include <torch/script.h>

#import <CoreML/CoreML.h>

#include <sys/utsname.h>
#include <fstream>
#include <iostream>

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
}

+ (void)setModelCacheDirectory:(NSString*)dir {
  gModelCacheDirectory = dir;
}

+ (NSString*)modelCacheDirectory {
  if (gModelCacheDirectory.length == 0) {
    // set the default directory to temp
    gModelCacheDirectory =
        [NSURL fileURLWithPath:NSTemporaryDirectory() isDirectory:YES].path;
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
  _modelPath =
      [self _saveModel:modelSpecs
            identifier:[NSString stringWithCString:identifier.c_str()
                                          encoding:NSUTF8StringEncoding]];
  NSError* error = nil;
  NSURL* compiledModelPath = [NSURL
      fileURLWithPath:[NSString stringWithFormat:@"%@.mlmodelc", _modelPath]];
  // Compile the model when OS version changes
  if ([self _shouldRecompileModel]) {
    if (@available(iOS 11.0, macOS 10.13, *)) {
      NSURL* temporaryFileURL =
          [MLModel compileModelAtURL:[NSURL URLWithString:_modelPath]
                               error:&error];
      // move the model to the cache directory
      NSFileManager* fileManager = [NSFileManager defaultManager];
      if ([fileManager fileExistsAtPath:compiledModelPath.path]) {
        [fileManager removeItemAtURL:compiledModelPath error:&error];
      }
      [fileManager moveItemAtURL:temporaryFileURL
                           toURL:compiledModelPath
                           error:&error];
    } else {
      TORCH_CHECK(false, "CoreML is not available on your deivce");
    }
  }

  if (error) {
    // remove cached models if compalition failed.
    [self cleanup];
    TORCH_CHECK(
        false,
        "Error compiling the MLModel",
        [error localizedDescription].UTF8String);
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
    _mlModel = [MLModel modelWithContentsOfURL:compiledModelPath
                                 configuration:config
                                         error:&error];
  } else {
    _mlModel = [MLModel modelWithContentsOfURL:compiledModelPath error:&error];
  }
  if (error || !_mlModel) {
    TORCH_CHECK(
        false,
        "Error loading the MLModel",
        error.localizedDescription.UTF8String);
  }

  _compiledModelPath = compiledModelPath.path;
  return YES;
}

- (id<MLFeatureProvider>)forwardWithInputs:
    (const std::vector<PTMCoreMLFeatureSpecs>&)inputs {
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
    if (error) {
      TORCH_CHECK(
          false,
          "Error running the prediction",
          error.localizedDescription.UTF8String);
    }

    return outputFeature;
  } else {
    TORCH_CHECK(false, "Core ML is not available on your device");
    return nil;
  }
}

- (BOOL)cleanup {
  NSFileManager* fileManager = [NSFileManager defaultManager];
  NSError* error = nil;
  if (![fileManager fileExistsAtPath:_modelPath]) {
    [fileManager removeItemAtPath:_modelPath error:&error];
  }
  if (![fileManager fileExistsAtPath:_compiledModelPath]) {
    [fileManager removeItemAtPath:_compiledModelPath error:&error];
  }
  return !error;
}

- (NSString*)_saveModel:(const std::string&)spec
             identifier:(NSString*)identifier {
  NSString* modelPath =
      [NSString stringWithFormat:@"%@/%@",
                                 [[self class] modelCacheDirectory],
                                 identifier];
  if (![[NSFileManager defaultManager] fileExistsAtPath:modelPath]) {
    // Note that the serialized protobuf binary contains bytes, not text;
    // see
    // https://developers.google.com/protocol-buffers/docs/pythontutorial#parsing-and-serialization
    NSData* data = [NSData dataWithBytes:spec.c_str() length:spec.length()];
    BOOL ret = [data writeToFile:modelPath atomically:YES];
    TORCH_CHECK(ret, "Error saving the MLModel", modelPath.UTF8String);
  }
  return modelPath;
}

- (BOOL)_shouldRecompileModel {
  NSError* error = nil;
  NSString* currentOSVer = [UIDevice currentDevice].systemVersion;
  NSString* versionPath = [NSString
      stringWithFormat:@"%@/version.txt", [[self class] modelCacheDirectory]];
  BOOL shouldRecompileModel = YES;
  NSFileManager* fileManager = [NSFileManager defaultManager];
  if ([fileManager fileExistsAtPath:versionPath]) {
    NSString* cachedOSVer =
        [NSString stringWithContentsOfFile:versionPath
                                  encoding:NSUTF8StringEncoding
                                     error:&error];
    if ([cachedOSVer isEqualToString:currentOSVer]) {
      NSString* compiledModelPath =
          [NSString stringWithFormat:@"%@.mlmodelc", _modelPath];
      if ([fileManager fileExistsAtPath:compiledModelPath]) {
        shouldRecompileModel = NO;
      }
    }
  }
  [currentOSVer writeToFile:versionPath atomically:YES];

  return shouldRecompileModel;
}

@end
