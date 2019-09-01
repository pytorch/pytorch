#import <Pytorch/Pytorch.h>
#import "TorchModule.h"
#import "TorchIValue+Internal.h"

@implementation TorchModule {
    std::shared_ptr<torch::jit::script::Module> _impl;
}

+ (TorchModule* _Nullable)loadTorchscriptModel:(NSString* _Nullable)modelPath {
    if(modelPath.length == 0){
        return nil;
    }
    auto torchScriptModule = torch::jit::load([modelPath cStringUsingEncoding:NSASCIIStringEncoding]);
    auto impl = std::make_shared<torch::jit::script::Module>(torchScriptModule);
    if (!impl) {
        return nil;
    }
    TorchModule* module = [TorchModule new];
    module->_impl = std::move(impl);
    return module;
}

- (TorchIValue* _Nullable)forward:(NSArray<TorchIValue* >* _Nullable)values {
    if (values.count == 0){
        return nil;
    }
    std::vector<at::IValue> inputs;
    for(TorchIValue* value in values) {
        at::IValue atValue = value.toIValue;
        inputs.push_back(atValue);
    }
    auto result = _impl->forward(inputs);
    return [TorchIValue newWithIValue:result];
}

- (TorchIValue* _Nullable)run_method:(NSString* _Nullable)methodName withInputs:(NSArray<TorchIValue* >* _Nullable) values {
    if (methodName.length == 0 || values.count ==0 ) {
        return nil;
    }
    std::vector<at::IValue> inputs;
    for(TorchIValue* value in values) {
        inputs.push_back(value.toIValue);
    }
    if (auto method = _impl->find_method(std::string([methodName cStringUsingEncoding:NSASCIIStringEncoding]))){
        auto result = (*method)(std::move(inputs));
        return [TorchIValue newWithIValue:result];
    }
    return nil;
}

@end
