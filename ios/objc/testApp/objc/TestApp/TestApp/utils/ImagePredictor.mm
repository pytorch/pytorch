#import "ImagePredictor.h"
#import <torch/script.h>
#import <PytorchObjC/PytorchObjC.h>
#import <vector>
#import "UIImage+Utils.h"

#define IMG_W 224
#define IMG_H 224
#define IMG_C 3

@implementation ImagePredictor {
    TorchModule* _module;
    NSArray* _labels;
}

- (instancetype)initWithModelPath:(NSString* )modelPath {
    self = [super init];
    if (self) {
        _module = [TorchModule loadTorchscriptModel:modelPath];
        NSError* err;
        NSString* str = [NSString stringWithContentsOfFile:[[NSBundle mainBundle] pathForResource:@"synset_words" ofType:@"txt"]
                                                  encoding:NSUTF8StringEncoding
                                                     error:&err];
        _labels = [str componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    }
    return self;
}

- (void)predict:(UIImage* )image
Completion:(void(^__nullable)(NSArray<NSDictionary* >* sortedResults))completion {
    
    float* pixels = [image resize:{IMG_W,IMG_H}].normalizedBuffer;
    TorchTensor* imageTensor = [TorchTensor newWithType:TorchTensorTypeFloat Size:@[ @(1), @(IMG_C), @(IMG_W), @(IMG_H) ] Data:(void* )pixels];
    TorchIValue* inputIValue = [TorchIValue newWithTensor:imageTensor];
    TorchTensor* outputTensor = [[_module forward:@[inputIValue]] toTensor];
    //collect the top 5 results
    NSArray<NSDictionary* >* sortedResults = [self topN:5 fromResults:outputTensor];
    if(completion){
        completion(sortedResults);
    }
}

- (NSArray<NSDictionary* >* )topN:(NSUInteger)k fromResults:(TorchTensor* ) results{
    int64_t totalCount = results.size[1].integerValue;
    NSMutableDictionary* scores = [NSMutableDictionary new];
    for(int i = 0; i<totalCount; ++i){
        scores[@(results[0][i].item.floatValue)] = @(i);
    }
    NSArray* keys = [scores allKeys];
    NSArray* sortedArray = [keys sortedArrayUsingComparator:^NSComparisonResult(NSNumber* obj1, NSNumber* obj2) {
        if (obj1.floatValue < obj2.floatValue){
            return NSOrderedDescending;
        }else if(obj1.floatValue > obj2.floatValue){
            return NSOrderedAscending;
        }else{
            return NSOrderedAscending;
        }
    }];
    NSMutableArray<NSDictionary* >* sortedResult = [NSMutableArray new];
    for(int i=0;i<k;i++){
        NSNumber* score = sortedArray[i];
        NSNumber* index = scores[score];
        NSString* label = _labels[index.integerValue];
        [sortedResult addObject:@{score:label}];
    }
    return sortedResult;
}

@end
