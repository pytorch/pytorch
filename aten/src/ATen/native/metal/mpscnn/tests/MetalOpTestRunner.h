// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#import <Foundation/Foundation.h>
#include <unordered_map>

@interface MetalOpTestRunner : NSObject

typedef BOOL(^testBlock)(void);

+ (instancetype)sharedInstance;
- (NSDictionary<NSString *, testBlock> *)tests;

@end
