#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLCompiler.h>

#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#endif

@implementation PTMCoreMLCompiler

static NSString *gCacheDirectory = @"";
static NSString *gCompiledModelExtension = @"mlmodelc";
static NSString *gVersionExtension = @"version";

+ (void)setCacheDirectory:(const std::string&)dir {
  gCacheDirectory = [NSString stringWithCString:dir.c_str()];
}

+ (nonnull NSString *)cacheDirectory {
  BOOL isSet = gCacheDirectory.length != 0;
  BOOL isWriteable = isSet && [[NSFileManager defaultManager] isWritableFileAtPath:gCacheDirectory];
  if (!isSet || !isWriteable) {
    // set the default directory to tmp
    gCacheDirectory = NSTemporaryDirectory();
  }
  return gCacheDirectory;
}

+ (BOOL)compileModel:(const std::string&)modelSpecs modelID:(const std::string&)modelID {
  NSString *modelName = [NSString stringWithCString:modelID.c_str() encoding:NSUTF8StringEncoding];
  NSString *modelPath = [NSTemporaryDirectory() stringByAppendingPathComponent:modelName];
  NSURL *compiledURL = [PTMCoreMLCompiler _cacheURLForModel:modelName extension:gCompiledModelExtension];
  BOOL compiledModelIsCached = [[NSFileManager defaultManager] fileExistsAtPath:compiledURL.path];

#if TARGET_OS_IPHONE
  NSURL *versionURL = [PTMCoreMLCompiler _cacheURLForModel:modelName extension:gVersionExtension];
  NSString *compilationOS = [NSString stringWithContentsOfFile:versionURL.path encoding:NSUTF8StringEncoding error:nil];
  NSString *currentOS = [UIDevice currentDevice].systemVersion;
  BOOL wasCachedOnThisOS = [currentOS isEqualToString:compilationOS];
#else
  BOOL wasCachedOnThisOS = NO;
#endif

  if (compiledModelIsCached && wasCachedOnThisOS) {
    return YES;
  }

  if (!wasCachedOnThisOS) {
    [PTMCoreMLCompiler _cleanupCachedModel:modelName];
  }

  BOOL writeSuccess = [PTMCoreMLCompiler _writeModelSpecs:modelSpecs toPath:modelPath];
  if (!writeSuccess) {
    return NO;
  }

  return [PTMCoreMLCompiler _compileModel:modelName atPath:modelPath];
}

+ (nullable MLModel*)loadModel:(const std::string)modelID backend:(const std::string)backend allowLowPrecision:(BOOL)allowLowPrecision error:(NSError**)error {
  NSString *modelName = [NSString stringWithCString:modelID.c_str() encoding:NSUTF8StringEncoding];
  NSURL *modelURL = [PTMCoreMLCompiler _cacheURLForModel:modelName extension:gCompiledModelExtension];

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
    model = [MLModel modelWithContentsOfURL:modelURL configuration:config error:error];
  } else {
    model = [MLModel modelWithContentsOfURL:modelURL error:error];
  }

  if (error && *error) {
    [PTMCoreMLCompiler _cleanupCachedModel:modelName];
    return nil;
  }

  return model;
}

+ (BOOL)_writeModelSpecs:(const std::string&)modelSpecs toPath:(NSString *)modelPath {
  // Note that the serialized protobuf binary contains bytes not text.
  // https://developers.google.com/protocol-buffers/docs/pythontutorial#parsing-and-serialization
  NSData* data = [NSData dataWithBytes:modelSpecs.c_str() length:modelSpecs.length()];
  return [data writeToFile:modelPath atomically:YES];
}

+ (BOOL)_compileModel:(NSString *)modelName atPath:(NSString *)modelPath {
  NSError *error;
  NSURL *modelURL = [NSURL fileURLWithPath:modelPath];
  NSURL *temporaryURL = [MLModel compileModelAtURL:modelURL error:&error];

  // After the compiled model has been created, the original specs can be cleared to save cache space.
  [[NSFileManager defaultManager] removeItemAtPath:modelPath error:nil];

  if (error) {
    return NO; // Model could not be compiled
  }

  NSURL *compiledURL = [PTMCoreMLCompiler _cacheURLForModel:modelName extension:gCompiledModelExtension];
  if (![compiledURL isEqual:temporaryURL]) {
    [[NSFileManager defaultManager] removeItemAtURL:compiledURL error:nil];
    [[NSFileManager defaultManager] moveItemAtURL:temporaryURL toURL:compiledURL error:&error];
  }

  if (error) {
    return NO; // Model could not be saved in cache
  }

#if TARGET_OS_IPHONE
  NSURL *versionURL = [PTMCoreMLCompiler _cacheURLForModel:modelName extension:gVersionExtension];
  NSString *currentOSVer = [UIDevice currentDevice].systemVersion;
  [currentOSVer writeToFile:versionURL.path atomically:YES];
#endif

  return YES;
}

+ (void)_cleanupCachedModel:(NSString *)modelName {
  NSURL *modelURL = [PTMCoreMLCompiler _cacheURLForModel:modelName extension:gCompiledModelExtension];
  NSURL *versionURL = [PTMCoreMLCompiler _cacheURLForModel:modelName extension:gVersionExtension];
  [[NSFileManager defaultManager] removeItemAtPath:modelURL.path error:nil];
  [[NSFileManager defaultManager] removeItemAtPath:versionURL.path error:nil];
}

+ (NSURL *)_cacheURLForModel:(NSString *)modelID extension:(NSString *)pathExtension {
  NSString *filename = [modelID stringByAppendingPathExtension:pathExtension];
  NSString *filePath = [[PTMCoreMLCompiler cacheDirectory] stringByAppendingPathComponent:filename];
  return [NSURL fileURLWithPath:filePath isDirectory:NO];
}

@end
