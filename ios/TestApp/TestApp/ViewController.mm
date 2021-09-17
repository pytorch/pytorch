#import "ViewController.h"

#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/script.h>


@interface ViewController ()
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    NSLog(@"a");
    NSString* modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"model_lite" ofType:@"ptl"];
    c10::InferenceMode mode;
    auto module = torch::jit::_load_for_mobile(modelPath.UTF8String);
    auto input = torch::ones({1, 3, 224, 224}, at::kFloat);
    auto outputTensor = module.forward({input}).toTensor();
    for(int i=0; i<20; ++i){
        std::cout<<outputTensor.data_ptr<float>()[i]<<std::endl;
    }
}

@end
