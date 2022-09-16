#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLCompiler.h>

#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#endif

@implementation PTMCoreMLCompiler

static NSString* gModelCacheDirectory = @"";

+ (void)setModelCacheDirectory:(const std::string&)dir {
  gModelCacheDirectory = [NSString stringWithCString:dir.c_str()];
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

+ (BOOL)compileModel:(const std::string&)modelSpecs modelID:(const std::string&)modelID {
  NSString* modelName = [NSString stringWithCString:modelID.c_str() encoding:NSUTF8StringEncoding];
  NSURL* modelPath = [PTMCoreMLCompiler _cacheFilePath:modelName];
  NSURL* compiledModelPath = [modelPath URLByAppendingPathExtension:@"mlmodelc"];

  BOOL modelIsCached = [[NSFileManager defaultManager] fileExistsAtPath:modelPath.path];
  BOOL compiledModelIsCached = [[NSFileManager defaultManager] fileExistsAtPath:compiledModelPath.path];

#if TARGET_OS_IPHONE
  NSError *error = nil;
  NSURL *compilationOSPath = [modelPath URLByAppendingPathExtension:@"version"];
  NSString *compilationOS = [NSString stringWithContentsOfFile:compilationOSPath.path encoding:NSUTF8StringEncoding error:&error];
  NSString *currentOS = [UIDevice currentDevice].systemVersion;
  BOOL wasCachedOnThisOS = [currentOS isEqualToString:compilationOS];
#else
  BOOL wasCachedOnThisOS = NO;
#endif

  if (modelIsCached != compiledModelIsCached || !wasCachedOnThisOS) {
    modelIsCached = NO;
    compiledModelIsCached = NO;
    [PTMCoreMLCompiler _cleanupCachedModel:modelID];
  }

  if (!modelIsCached) {
    // Note that the serialized protobuf binary contains bytes not text.
    // https://developers.google.com/protocol-buffers/docs/pythontutorial#parsing-and-serialization
    NSData* data = [NSData dataWithBytes:modelSpecs.c_str() length:modelSpecs.length()];
    if (![data writeToFile:modelPath.path atomically:YES]) {
      // If the model cannot be persisted on disk then compilation cannot proceed.
      NSLog(@"Failed to save specs for MLModel!");
      [PTMCoreMLCompiler _cleanupCachedModel:modelID];
      return NO;
    }
  }

  if (compiledModelIsCached) {
    return YES;
  }

  return [PTMCoreMLCompiler _compileModel:modelID atPath:modelPath andCache:compiledModelPath];
}

+ (nullable MLModel*)loadModel:(const std::string&)modelID backend:(const std::string&)backend allowLowPrecision:(BOOL)allowLowPrecision {
  NSString* modelName = [NSString stringWithCString:modelID.c_str() encoding:NSUTF8StringEncoding];
  NSURL* modelPath = [PTMCoreMLCompiler _cacheFilePath:modelName];
  NSURL* compiledModelPath = [modelPath URLByAppendingPathExtension:@"mlmodelc"];

  NSError *error;
  MLModel *model;
  if (@available(iOS 12.0, macOS 10.14, *)) {
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    MLComputeUnits computeUnits = MLComputeUnitsCPUOnly;
    if (backend == "cpuAndGPU") {
      computeUnits = MLComputeUnitsCPUAndGPU;
    } else if (backend == "all") {
      computeUnits = MLComputeUnitsAll;
    }
    config.computeUnits = computeUnits;
    config.allowLowPrecisionAccumulationOnGPU = allowLowPrecision;
    model = [MLModel modelWithContentsOfURL:compiledModelPath configuration:config error:&error];
  } else {
    model = [MLModel modelWithContentsOfURL:compiledModelPath error:&error];
  }

  if (error) {
    NSLog(@"Failed to initialize MLModel!");
    [PTMCoreMLCompiler _cleanupCachedModel:modelID];
    return nil;
  }

  return model;
}

+ (BOOL)_compileModel:(const std::string&)modelID atPath:(NSURL *)modelPath andCache:(NSURL *)cachePath {
  NSError *error;
  NSURL *temporaryURL = [MLModel compileModelAtURL:modelPath error:&error];
  if (!error) {
#if TARGET_OS_IPHONE
    NSURL *compilationOSPath = [modelPath URLByAppendingPathExtension:@"version"];
    NSString *currentOSVer = [UIDevice currentDevice].systemVersion;
    [currentOSVer writeToFile:compilationOSPath.path atomically:YES];
#endif
    [PTMCoreMLCompiler _moveFileToCache:temporaryURL cacheURL:cachePath error:&error];
  }
  if (error) {
    NSLog(@"Failed to compile MLModel!");
    [PTMCoreMLCompiler _cleanupCachedModel:modelID];
  }
  return !error;
}

+ (void)_cleanupCachedModel:(const std::string&)modelID {
  NSString* modelName = [NSString stringWithCString:modelID.c_str() encoding:NSUTF8StringEncoding];
  NSURL* modelPath = [PTMCoreMLCompiler _cacheFilePath:modelName];
  NSURL* compiledModelPath = [modelPath URLByAppendingPathExtension:@"mlmodelc"];
  NSURL* compilationOSPath = [modelPath URLByAppendingPathExtension:@"version"];
  NSError* error = nil;
  [[NSFileManager defaultManager] removeItemAtPath:modelPath.path error:&error];
  [[NSFileManager defaultManager] removeItemAtPath:compiledModelPath.path error:&error];
  [[NSFileManager defaultManager] removeItemAtPath:compilationOSPath.path error:&error];
}

+ (void)_moveFileToCache:(NSURL *)fileURL cacheURL:(NSURL *)cacheURL error:(NSError **)error {
  if ([fileURL isEqual:cacheURL]) {
    return;
  }
  [[NSFileManager defaultManager] removeItemAtURL:cacheURL error:nil];
  [[NSFileManager defaultManager] moveItemAtURL:fileURL toURL:cacheURL error:error];
}

+ (NSURL *)_cacheFilePath:(NSString *)fileName {
  NSString *filePath = [[PTMCoreMLCompiler modelCacheDirectory] stringByAppendingPathComponent:fileName];
  return [NSURL fileURLWithPath:filePath];
}

@end
