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

+ (NSURL*)compileModel:(const std::string&)modelSpecs modelID:(const std::string&)modelID {
  NSString* modelName = [NSString stringWithCString:modelID.c_str() encoding:NSUTF8StringEncoding];
  NSURL* modelPath = [PTMCoreMLCompiler _cacheFilePath:modelName];
  NSURL* compiledModelPath = [PTMCoreMLCompiler _compiledModelFilePath:modelPath.path];

  BOOL modelCached = [[NSFileManager defaultManager] fileExistsAtPath:modelPath.path];
  BOOL compiledModelCached = [[NSFileManager defaultManager] fileExistsAtPath:compiledModelPath.path];
  BOOL shouldRecompile = [self _shouldRecompileModel:compiledModelPath];

  if (modelCached != compiledModelCached) {
    modelCached = NO;
    compiledModelCached = NO;
    [PTMCoreMLCompiler _cleanupModel:modelPath compiledModel:compiledModelPath];
  }

  if (!modelCached) {
    // Note that the serialized protobuf binary contains bytes not text.
    // https://developers.google.com/protocol-buffers/docs/pythontutorial#parsing-and-serialization
    NSData* data = [NSData dataWithBytes:modelSpecs.c_str() length:modelSpecs.length()];
    if (![data writeToFile:modelPath.path atomically:YES]) {
        // If the model cannot be persisted on disk then compilation cannot proceed.
        NSLog(@"Failed to save specs for MLModel!");
        [PTMCoreMLCompiler _cleanupModel:modelPath compiledModel:compiledModelPath];
        return nil;
    }
  }

  if (shouldRecompile || !compiledModelCached) {
    NSError *error;
    NSURL *temporaryURL = [MLModel compileModelAtURL:modelPath error:&error];
    if (!error) {
      [PTMCoreMLCompiler _moveFileToCache:temporaryURL cacheURL:compiledModelPath error:&error];
    }
    if (error) {
      NSLog(@"Failed to compile MLModel!");
      [PTMCoreMLCompiler _cleanupModel:modelPath compiledModel:compiledModelPath];
      return nil;
    }
  }

  return compiledModelPath;
}

+ (nullable MLModel*)loadCPUModelAtURL:(NSURL*)modelURL {
  NSError *error;
  MLModel *model;
  if (@available(iOS 12.0, macOS 10.14, *)) {
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsCPUOnly;
    model = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&error];
  } else {
    model = [MLModel modelWithContentsOfURL:modelURL error:&error];
  }
  if (error) {
    NSLog(@"Failed to initialize MLModel!");
    [PTMCoreMLCompiler _cleanupModel:nil compiledModel:modelURL];
    return nil;
  }
  return model;
}

+ (nullable MLModel*)loadModelAtURL:(NSURL*)modelURL backend:(const std::string&)backend allowLowPrecision:(BOOL)allowLowPrecision {
  NSError *error;
  MLModel *model;
  if (@available(iOS 12.0, macOS 10.14, *)) {
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    MLComputeUnits computeUnits = MLComputeUnitsCPUOnly;
    if (backend == "cpuandgpu") {
      computeUnits = MLComputeUnitsCPUAndGPU;
    } else if (backend == "all") {
      computeUnits = MLComputeUnitsAll;
    }
    config.computeUnits = computeUnits;
    config.allowLowPrecisionAccumulationOnGPU = allowLowPrecision;
    model = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&error];
  } else {
    model = [MLModel modelWithContentsOfURL:modelURL error:&error];
  }
  if (error) {
    NSLog(@"Failed to initialize MLModel!");
    [PTMCoreMLCompiler _cleanupModel:nil compiledModel:modelURL];
    return nil;
  }
  return model;
}

+ (void)_cleanupModel:(NSURL*)modelPath compiledModel:(NSURL*)compiledModelPath {
  NSFileManager* fileManager = [NSFileManager defaultManager];
  NSError* error = nil;
  if (modelPath && [fileManager fileExistsAtPath:modelPath.path]) {
    [fileManager removeItemAtPath:modelPath.path error:&error];
  }
  if (compiledModelPath && [fileManager fileExistsAtPath:compiledModelPath.path]) {
    [fileManager removeItemAtPath:compiledModelPath.path error:&error];
  }
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
  NSString *versionPath = [PTMCoreMLCompiler _cacheFilePath:@"version"].path;
  NSString *cachedOSVer = nil;
  if ([[NSFileManager defaultManager] fileExistsAtPath:versionPath]) {
    NSError *error = nil;
    cachedOSVer = [NSString stringWithContentsOfFile:versionPath encoding:NSUTF8StringEncoding error:&error];
  }
  // Compile the model when OS version changes
  NSString *currentOSVer = [UIDevice currentDevice].systemVersion;
  [currentOSVer writeToFile:versionPath atomically:YES];
  return ![currentOSVer isEqualToString:cachedOSVer];
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
