// Copyright 2004-present Facebook. All Rights Reserved.

#import "arm_neon_support.h"
#import "MetalContext.h"
#import "FBMetalCNNNoOp.h"
#import "metal_sync_op.h"

static MetalContext* metalContext = NULL;
static FBMetalCNNNoOp* noOpCache = NULL;

static void init_metal_pipeline() {
  if (metalContext == NULL) {
    metalContext = [MetalContext getContext];
  }
}

bool metal_sync_op() {
  init_metal_pipeline();

  if (noOpCache == NULL) {
    noOpCache = [FBMetalCNNNoOp filterWithContext:metalContext
                                            width:1
                                           height:1];
  }

  static dispatch_semaphore_t gpu_execution_done = NULL;
  if (gpu_execution_done == NULL)
    gpu_execution_done = dispatch_semaphore_create(0);

  [noOpCache applyFilter:^(NSError* error) {
    dispatch_semaphore_signal(gpu_execution_done);
  }];
  dispatch_semaphore_wait(gpu_execution_done, DISPATCH_TIME_FOREVER);

  return true;
}
