/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*

Most of NeuralNetworks.h has been stripped for simplicity.
We don't need any of the function declarations since
we call them all through dlopen/dlsym.
Operation codes are pulled directly from serialized models.

*/

#ifndef MINIMAL_NEURAL_NETWORKS_H
#define MINIMAL_NEURAL_NETWORKS_H

#include <stdint.h>

typedef enum {
    ANEURALNETWORKS_NO_ERROR = 0,
    ANEURALNETWORKS_OUT_OF_MEMORY = 1,
    ANEURALNETWORKS_INCOMPLETE = 2,
    ANEURALNETWORKS_UNEXPECTED_NULL = 3,
    ANEURALNETWORKS_BAD_DATA = 4,
    ANEURALNETWORKS_OP_FAILED = 5,
    ANEURALNETWORKS_BAD_STATE = 6,
    ANEURALNETWORKS_UNMAPPABLE = 7,
    ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE = 8,
    ANEURALNETWORKS_UNAVAILABLE_DEVICE = 9,
} ResultCode;

typedef enum {
    ANEURALNETWORKS_FLOAT32 = 0,
    ANEURALNETWORKS_INT32 = 1,
    ANEURALNETWORKS_UINT32 = 2,
    ANEURALNETWORKS_TENSOR_FLOAT32 = 3,
    ANEURALNETWORKS_TENSOR_INT32 = 4,
    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM = 5,
    ANEURALNETWORKS_BOOL = 6,
    ANEURALNETWORKS_TENSOR_QUANT16_SYMM = 7,
    ANEURALNETWORKS_TENSOR_FLOAT16 = 8,
    ANEURALNETWORKS_TENSOR_BOOL8 = 9,
    ANEURALNETWORKS_FLOAT16 = 10,
    ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL = 11,
    ANEURALNETWORKS_TENSOR_QUANT16_ASYMM = 12,
    ANEURALNETWORKS_TENSOR_QUANT8_SYMM = 13,
} OperandCode;

typedef enum {
    ANEURALNETWORKS_PREFER_LOW_POWER = 0,
    ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1,
    ANEURALNETWORKS_PREFER_SUSTAINED_SPEED = 2,
} PreferenceCode;

typedef struct ANeuralNetworksMemory ANeuralNetworksMemory;
typedef struct ANeuralNetworksModel ANeuralNetworksModel;
typedef struct ANeuralNetworksDevice ANeuralNetworksDevice;
typedef struct ANeuralNetworksCompilation ANeuralNetworksCompilation;
typedef struct ANeuralNetworksExecution ANeuralNetworksExecution;
typedef struct ANeuralNetworksEvent ANeuralNetworksEvent;

typedef int32_t ANeuralNetworksOperationType;

typedef struct ANeuralNetworksOperandType {
    int32_t type;
    uint32_t dimensionCount;
    const uint32_t* dimensions;
    float scale;
    int32_t zeroPoint;
} ANeuralNetworksOperandType;

#endif  // MINIMAL_NEURAL_NETWORKS_H
