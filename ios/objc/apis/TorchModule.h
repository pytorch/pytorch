#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN


@class TorchIValue;
@interface TorchModule : NSObject

/**
 Load the torchscript model and returns a TorchModule object.
 see https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
 @param modelPath File path to the torchscript model
 @return TorchModuel object
 */
+ (TorchModule* _Nullable)loadTorchscriptModel:(NSString* _Nullable)modelPath;

/**
 Run the default inference method in the torchscript model
 @param values A list of IValue type objects
 @return The inference result
 */
- (TorchIValue* _Nullable)forward:(NSArray<TorchIValue* >* _Nullable)values;

/**
 Run a method defined in the torchscript model
 @param methodName The name of the methods in the model
 @param inputs A list of IValue type objects
 @return The inference result
 */
- (TorchIValue* _Nullable)run_method:(NSString* _Nullable)methodName withInputs:(NSArray<TorchIValue* >* _Nullable) inputs;

@end

NS_ASSUME_NONNULL_END
