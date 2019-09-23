#import <LibTorch/LibTorch.h>
#import "TorchModule.h"
#import "TorchIValuePrivate.h"

@implementation TorchModule {
  torch::jit::script::Module _impl;
}

+ (TorchModule*)loadTorchscriptModel:(NSString*)modelPath {
  if (modelPath.length == 0) {
    return nil;
  }
  @try {
    auto torchScriptModule =
        torch::jit::load([modelPath cStringUsingEncoding:NSASCIIStringEncoding]);
    TorchModule* module = [[TorchModule alloc] init];
    module->_impl = torchScriptModule;
    return module;
  } @catch (NSException* exception) {
    @throw exception;
    NSLog(@"%@", exception);
  }
  return nil;
}

+ (NSArray<NSNumber* >* )predict:(void* )data dims:(NSArray<NSNumber* >* )dims type:(TensorType) type {
    return nil;
}

- (TorchIValue*)forward:(NSArray<TorchIValue*>*)values {
  std::vector<at::IValue> inputs;
  for (TorchIValue* value in values) {
    at::IValue atValue = value.toIValue;
    inputs.push_back(atValue);
  }
  @try {
    torch::autograd::AutoGradMode guard(false);
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    auto result = _impl.forward(inputs);
    return [TorchIValue newWithIValue:result];
  } @catch (NSException* exception) {
    @throw exception;
    NSLog(@"%@", exception);
  }
  return nil;
}

- (TorchIValue*)run_method:(NSString*)methodName withInputs:(NSArray<TorchIValue*>*)values {
  if (methodName.length == 0) {
    return nil;
  }
  std::vector<at::IValue> inputs;
  for (TorchIValue* value in values) {
    inputs.push_back(value.toIValue);
  }
  @try {
    if (auto method = _impl.find_method(
            std::string([methodName cStringUsingEncoding:NSASCIIStringEncoding]))) {
      torch::autograd::AutoGradMode guard(false);
      at::AutoNonVariableTypeMode non_var_type_mode(true);
      auto result = (*method)(std::move(inputs));
      return [TorchIValue newWithIValue:result];
    }
  } @catch (NSException* exception) {
    @throw exception;
    NSLog(@"%@", exception);
  }
  return nil;
}

@end
