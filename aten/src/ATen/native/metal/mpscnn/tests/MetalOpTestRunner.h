// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <c10/macros/Macros.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-declarations")
#import <Foundation/Foundation.h>
C10_DIAGNOSTIC_POP()
#include <unordered_map>

@interface MetalOpTestRunner : NSObject

typedef BOOL(^testBlock)(void);

+ (instancetype)sharedInstance;
- (NSDictionary<NSString *, testBlock> *)tests;

@end
