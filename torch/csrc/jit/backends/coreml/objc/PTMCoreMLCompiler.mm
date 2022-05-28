#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLCompiler.h>
#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLError.h>

#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#endif

@implementation PTMCoreMLCompiler

static NSString* gModelCacheDirectory = @"";

+ (void)setModelCacheDirectory:(nonnull NSString *)dir {
  gModelCacheDirectory = dir;
}

+ (nonnull NSString *)modelCacheDirectory {
  BOOL isSet = gModelCacheDirectory.length != 0;
  BOOL isWriteable = isSet && [[NSFileManager defaultManager] isWritableFileAtPath:gModelCacheDirectory];
  if (!isSet || !isWriteable) {
    // set the default directory to tmp
    gModelCacheDirectory = NSTemporaryDirectory();
  }
  return gModelCacheDirectory;
}

+ (nullable NSDictionary *)parseSpecs:(const char *)json error:(NSError **)error {
  NSString *jsonStr = [[NSString alloc] initWithCString:json encoding:NSUTF8StringEncoding];
  NSData *data = [jsonStr dataUsingEncoding:NSUTF8StringEncoding];
  id serializedObject = [NSJSONSerialization JSONObjectWithData:data options:0 error:error];
  if (![serializedObject isKindOfClass:[NSDictionary class]]) {
    return nil;
  }
  return (NSDictionary *)serializedObject;
}

+ (nullable MLModel *)compileMLModel:(const std::string&)modelSpecs
                          identifier:(const std::string&)identifier
                             backend:(NSString *)backend
                   allowLowPrecision:(BOOL)allowLowPrecision
                               error:(NSError **)error {
  if (!@available(iOS 11.0, macOS 10.13, *)) {
    *error = [NSError errorWithDomain:kPTMCoreMLErrorDomain code:PTMCoreMLErrorCodeOSVersion userInfo:nil];
    return nil;
  }

  NSString* modelName = [NSString stringWithCString:identifier.c_str() encoding:NSUTF8StringEncoding];
  NSURL* modelPath = [PTMCoreMLCompiler _cacheFilePath:modelName];
  NSURL* compiledModelPath = [PTMCoreMLCompiler _compiledModelFilePath:modelPath.path];

  BOOL modelSaved = [PTMCoreMLCompiler _saveModel:modelSpecs path:modelPath];

  if (!modelSaved) {
    // If the model cannot be persisted on disk then compilation cannot proceed.
    *error = [NSError errorWithDomain:kPTMCoreMLErrorDomain code:PTMCoreMLErrorCodeFailedSave userInfo:nil];
    return nil;
  }

  [PTMCoreMLCompiler _recompileIfNeeded:compiledModelPath compiledModelPath:compiledModelPath error:error];

  if (*error) {
    [PTMCoreMLCompiler _cleanupModel:modelPath compiledModel:compiledModelPath];
    return nil;
  }

  MLModel *compiledModel;

  if (@available(iOS 12.0, macOS 10.14, *)) {
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    MLComputeUnits computeUnits = MLComputeUnitsCPUOnly;
    if ([backend isEqualToString:@"cpuandgpu"]) {
      computeUnits = MLComputeUnitsCPUAndGPU;
    } else if ([backend isEqualToString:@"all"]) {
      computeUnits = MLComputeUnitsAll;
    }
    config.computeUnits = computeUnits;
    config.allowLowPrecisionAccumulationOnGPU = allowLowPrecision;
    compiledModel = [MLModel modelWithContentsOfURL:compiledModelPath configuration:config error:error];
  } else {
    compiledModel = [MLModel modelWithContentsOfURL:compiledModelPath error:error];
  }

  if (!compiledModel && *error == nil) {
    *error = [NSError errorWithDomain:kPTMCoreMLErrorDomain code:PTMCoreMLErrorCodeUnknown userInfo:nil];
  }

  return compiledModel;
}

+ (void)_cleanupModel:(NSURL*)modelPath compiledModel:(NSURL*)compiledModelPath {
  NSFileManager* fileManager = [NSFileManager defaultManager];
  NSError* error = nil;
  if ([fileManager fileExistsAtPath:modelPath.path]) {
    [fileManager removeItemAtPath:modelPath.path error:&error];
  }
  if ([fileManager fileExistsAtPath:compiledModelPath.path]) {
    [fileManager removeItemAtPath:compiledModelPath.path error:&error];
  }
}

+ (BOOL)_saveModel:(const std::string&)spec path:(NSURL*)modelPath {
  if ([[NSFileManager defaultManager] fileExistsAtPath:modelPath.path]) {
    return YES;
  }
  // Note that the serialized protobuf binary contains bytes not text.
  // https://developers.google.com/protocol-buffers/docs/pythontutorial#parsing-and-serialization
  NSData* data = [NSData dataWithBytes:spec.c_str() length:spec.length()];
  return [data writeToFile:modelPath.path atomically:YES];
}

+ (void)_recompileIfNeeded:(NSURL*)modelPath compiledModelPath:(NSURL*)compiledModelPath error:(NSError **)error {
  if (![PTMCoreMLCompiler _shouldRecompileModel:compiledModelPath]) {
    return;
  }

  NSURL *temporaryURL = [MLModel compileModelAtURL:modelPath error:error];
  if (*error) {
    return;
  }

  [PTMCoreMLCompiler _moveFileToCache:temporaryURL cacheURL:compiledModelPath error:error];
}

+ (void)_moveFileToCache:(NSURL *)fileURL cacheURL:(NSURL *)cacheURL error:(NSError **)error {
  if ([fileURL isEqual:cacheURL]) {
    return;
  }
  NSFileManager *fileManager = [NSFileManager defaultManager];
  if ([fileManager fileExistsAtPath:cacheURL.path]) {
    [fileManager removeItemAtURL:cacheURL error:error];
  }
  [fileManager moveItemAtURL:fileURL toURL:cacheURL error:error];
}

+ (BOOL)_shouldRecompileModel:(NSURL *)compiledModelPath {
#if TARGET_OS_IPHONE
  NSString *currentOSVer = [UIDevice currentDevice].systemVersion;
  NSString *versionPath = [PTMCoreMLCompiler _cacheFilePath:@"version"].path;
  BOOL cachedOSExists = [[NSFileManager defaultManager] fileExistsAtPath:versionPath];
  if (!cachedOSExists) {
    [currentOSVer writeToFile:versionPath atomically:YES];
    return YES;
  }
  // Compile the model when OS version changes
  NSError *error = nil;
  NSString *cachedOSVer = [NSString stringWithContentsOfFile:versionPath encoding:NSUTF8StringEncoding error:&error];
  BOOL changedOS = ![cachedOSVer isEqualToString:currentOSVer];
  BOOL compiledModelExists = !changedOS && [[NSFileManager defaultManager] fileExistsAtPath:compiledModelPath.path];
  [currentOSVer writeToFile:versionPath atomically:YES];
  return !compiledModelExists;
#else
  return YES;
#endif
}

+ (NSURL *)_cacheFilePath:(NSString *)fileName {
  NSString *filePath = [[PTMCoreMLCompiler modelCacheDirectory] stringByAppendingPathComponent:fileName];
  return [NSURL fileURLWithPath:filePath];
}

+ (NSURL *)_compiledModelFilePath:(NSString *)modelPath {
  NSString *filePath = [modelPath stringByAppendingString:@".mlmodelc"];
  return [NSURL fileURLWithPath:filePath];
}

@end
