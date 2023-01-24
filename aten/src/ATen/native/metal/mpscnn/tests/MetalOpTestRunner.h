// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#import <Foundation/Foundation.h>
#include <unordered_map>

@interface MetalOpTestRunner : NSObject

typedef BOOL(^testBlock)(void);

+ (instancetype)sharedInstance;
- (NSDictionary<NSString *, testBlock> *)tests;

@end
