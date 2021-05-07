// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#import <Foundation/Foundation.h>

@interface MetalOpTestRunner : NSObject

// result dict
// key: test name
// value: rest result (@(1): succeeded, @(0): failed)
+ (NSDictionary<NSString *, NSNumber *> *)testMPSCNNOps;

@end
