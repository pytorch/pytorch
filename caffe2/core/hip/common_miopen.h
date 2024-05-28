/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef CAFFE2_CORE_COMMON_MIOPEN_H_
#define CAFFE2_CORE_COMMON_MIOPEN_H_

#include <array>
#include <mutex>
#include "miopen/miopen.h"
#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2_pb.h"

#define MIOPEN_VERSION 1399

namespace caffe2 {

namespace internal {
/**
 * A helper function to obtain miopen error strings.
 */
inline const char* miopenGetErrorString(miopenStatus_t status)
{
    switch(status)
    {
    case miopenStatusSuccess: return "MIOPEN_STATUS_SUCCESS";
    case miopenStatusNotInitialized: return "MIOPEN_STATUS_NOT_INITIALIZED";
    case miopenStatusAllocFailed: return "MIOPEN_STATUS_ALLOC_FAILED";
    case miopenStatusBadParm: return "MIOPEN_STATUS_BAD_PARAM";
    case miopenStatusInternalError: return "MIOPEN_STATUS_INTERNAL_ERROR";
    case miopenStatusInvalidValue: return "MIOPEN_STATUS_INVALID_VALUE";
    case miopenStatusNotImplemented: return "MIOPEN_STATUS_NOT_SUPPORTED";
    case miopenStatusUnknownError: return "MIOPEN_STATUS_UNKNOWN_ERROR";
    default: return "MIOPEN_STATUS_UNKNOWN_ERROR";
    }
}
} // namespace internal

// A macro that wraps around a miopen statement so we can check if the miopen
// execution finishes or not.
#define MIOPEN_ENFORCE(condition)                                           \
    do                                                                      \
    {                                                                       \
        miopenStatus_t status = condition;                                  \
        CAFFE_ENFORCE_EQ(status,                                            \
                         miopenStatusSuccess,                               \
                         ", Error at: ",                                    \
                         __FILE__,                                          \
                         ":",                                               \
                         __LINE__,                                          \
                         ": ",                                              \
                         ::caffe2::internal::miopenGetErrorString(status)); \
    } while(0)
#define MIOPEN_CHECK(condition)                                                                   \
    do                                                                                            \
    {                                                                                             \
        miopenStatus_t status = condition;                                                        \
        CHECK(status == miopenStatusSuccess) << ::caffe2::internal::miopenGetErrorString(status); \
    } while(0)

// report the version of miopen Caffe2 was compiled with
inline size_t miopenCompiledVersion() { return MIOPEN_VERSION; }

// report the runtime version of miopen
inline size_t miopenRuntimeVersion() { return MIOPEN_VERSION; }

// Check compatibility of compiled and runtime miopen versions
inline void CheckMIOPENVersions() {}

/**
 * miopenTypeWrapper is a wrapper class that allows us to refer to the miopen type
 * in a template function. The class is specialized explicitly for different
 * data types below.
 */
template <typename T>
class miopenTypeWrapper;

template <>
class miopenTypeWrapper<float>
{
    public:
    static const miopenDataType_t type = miopenFloat;
    typedef const float ScalingParamType;
    typedef float BNParamType;
    static ScalingParamType* kOne()
    {
        static ScalingParamType v = 1.0;
        return &v;
    }
    static const ScalingParamType* kZero()
    {
        static ScalingParamType v = 0.0;
        return &v;
    }
};

template <>
class miopenTypeWrapper<at::Half>
{
    public:
    static const miopenDataType_t type = miopenHalf;
    typedef const float ScalingParamType;
    typedef float BNParamType;
    static ScalingParamType* kOne()
    {
        static ScalingParamType v = 1.0;
        return &v;
    }
    static ScalingParamType* kZero()
    {
        static ScalingParamType v = 0.0;
        return &v;
    }
};

/**
 * miopenTensorDescWrapper is the placeholder that wraps around a
 * miopenTensorDescriptor_t, allowing us to do descriptor change as-needed during
 * runtime.
 */
class miopenTensorDescWrapper
{
    public:
    miopenTensorDescWrapper() { MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&desc_)); }
    ~miopenTensorDescWrapper() noexcept { MIOPEN_CHECK(miopenDestroyTensorDescriptor(desc_)); }

    inline miopenTensorDescriptor_t
    Descriptor(const miopenDataType_t type, const vector<int>& dims, bool* changed)
    {
        if(type_ == type && dims_ == dims)
        {
            // if not changed, simply return the current descriptor.
            if(changed)
                *changed = false;
            return desc_;
        }
        CAFFE_ENFORCE_EQ(
            dims.size(), 4, "MIOPEN currently only support 4-dimensional tensor descriptor");

        type_ = type;
        dims_ = dims;
        MIOPEN_ENFORCE(
            miopenSet4dTensorDescriptor(desc_, type, dims_[0], dims_[1], dims_[2], dims_[3]));
        if(changed)
            *changed = true;
        return desc_;
    }

    template <typename T>
    inline miopenTensorDescriptor_t Descriptor(const StorageOrder& order, const vector<int>& dims)
    {
        return Descriptor(miopenTypeWrapper<T>::type, dims, nullptr);
    }

    private:
    miopenTensorDescriptor_t desc_;
    miopenDataType_t type_;
    vector<int> dims_;
    C10_DISABLE_COPY_AND_ASSIGN(miopenTensorDescWrapper);
};

} // namespace caffe2

#endif // CAFFE2_CORE_COMMON_MIOPEN_H_
