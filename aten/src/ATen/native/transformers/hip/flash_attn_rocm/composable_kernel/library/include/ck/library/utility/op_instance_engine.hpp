// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "ck/utility/functional2.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"

namespace ck {
namespace utils {

struct ProfileBestConfig
{
    std::string best_op_name;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_tflops     = std::numeric_limits<float>::max();
    float best_gb_per_sec = std::numeric_limits<float>::max();
};

/**
 * @brief      This class describes an operation instance(s).
 *
 *             Op instance defines a particular specializations of operator
 *             template. Thanks to this specific input/output data types, data
 *             layouts and modifying elementwise operations it is able to create
 *             it's input/output tensors, provide pointers to instances which
 *             can execute it and all operation specific parameters.
 */
template <typename OutDataType, typename... InArgTypes>
class OpInstance
{
    public:
    template <typename T>
    using TensorPtr      = std::unique_ptr<Tensor<T>>;
    using InTensorsTuple = std::tuple<TensorPtr<InArgTypes>...>;
    using DeviceMemPtr   = std::unique_ptr<DeviceMem>;
    using DeviceBuffers  = std::vector<DeviceMemPtr>;

    OpInstance()                  = default;
    OpInstance(const OpInstance&) = default;
    OpInstance& operator=(const OpInstance&) = default;
    virtual ~OpInstance(){};

    virtual InTensorsTuple GetInputTensors() const         = 0;
    virtual TensorPtr<OutDataType> GetOutputTensor() const = 0;
    virtual std::unique_ptr<tensor_operation::device::BaseInvoker>
    MakeInvokerPointer(tensor_operation::device::BaseOperator*) const = 0;
    virtual std::unique_ptr<tensor_operation::device::BaseArgument>
    MakeArgumentPointer(tensor_operation::device::BaseOperator*,
                        const DeviceBuffers&,
                        const DeviceMemPtr&) const = 0;
    virtual std::size_t GetFlops() const           = 0;
    virtual std::size_t GetBtype() const           = 0;
};

/**
 * @brief      A generic operation instance run engine.
 */
template <typename OutDataType, typename... InArgTypes>
class OpInstanceRunEngine
{
    public:
    using OpInstanceT = OpInstance<InArgTypes..., OutDataType>;
    template <typename T>
    using TensorPtr        = std::unique_ptr<Tensor<T>>;
    using DeviceMemPtr     = std::unique_ptr<DeviceMem>;
    using InTensorsTuple   = std::tuple<TensorPtr<InArgTypes>...>;
    using DeviceBuffers    = std::vector<DeviceMemPtr>;
    using InArgsTypesTuple = std::tuple<InArgTypes...>;

    OpInstanceRunEngine() = delete;

    template <typename ReferenceOp = std::function<void()>>
    OpInstanceRunEngine(const OpInstanceT& op_instance,
                        const ReferenceOp& reference_op = ReferenceOp{},
                        bool do_verification            = true)
        : op_instance_{op_instance}
    {
        in_tensors_ = op_instance_.GetInputTensors();
        out_tensor_ = op_instance_.GetOutputTensor();

        if constexpr(std::is_invocable_v<ReferenceOp,
                                         const Tensor<InArgTypes>&...,
                                         Tensor<OutDataType>&>)
        {
            if(do_verification)
            {
                ref_output_ = op_instance_.GetOutputTensor();
                CallRefOpUnpackArgs(reference_op, std::make_index_sequence<kNInArgs_>{});
            }
        }
        AllocateDeviceInputTensors(std::make_index_sequence<kNInArgs_>{});
        out_device_buffer_ = std::make_unique<DeviceMem>(sizeof(OutDataType) *
                                                         out_tensor_->mDesc.GetElementSpaceSize());
        out_device_buffer_->SetZero();
    }

    virtual ~OpInstanceRunEngine(){};

    template <typename OpInstancePtr>
    bool Test(const std::vector<OpInstancePtr>& op_ptrs)
    {
        bool res{true};
        for(auto& op_ptr : op_ptrs)
        {
            auto invoker  = op_instance_.MakeInvokerPointer(op_ptr.get());
            auto argument = op_instance_.MakeArgumentPointer(
                op_ptr.get(), in_device_buffers_, out_device_buffer_);
            if(op_ptr->IsSupportedArgument(argument.get()))
            {
                std::cout << "Testing instance: " << op_ptr->GetTypeString() << std::endl;
                invoker->Run(argument.get());
                out_device_buffer_->FromDevice(out_tensor_->mData.data());
                if(!ref_output_)
                {
                    throw std::runtime_error(
                        "OpInstanceRunEngine::Test: Reference value not availabe."
                        " You have to provide reference function.");
                }
                // TODO: enable flexible use of custom check_error functions
                bool inst_res = CheckErr(out_tensor_->mData, ref_output_->mData);
                std::cout << (inst_res ? "SUCCESS" : "FAILURE") << std::endl;
                res = res && inst_res;
                out_device_buffer_->SetZero();
            }
            else
            {
                std::cout << "Given conv problem is not supported by instance: \n\t>>>>"
                          << op_ptr->GetTypeString() << std::endl;
            }
        }
        return res;
    }

    template <typename OpInstancePtr>
    ProfileBestConfig Profile(const std::vector<OpInstancePtr>& op_ptrs,
                              bool time_kernel     = false,
                              bool do_verification = false,
                              bool do_log          = false)
    {
        ProfileBestConfig best_config;

        for(auto& op_ptr : op_ptrs)
        {
            auto invoker  = op_instance_.MakeInvokerPointer(op_ptr.get());
            auto argument = op_instance_.MakeArgumentPointer(
                op_ptr.get(), in_device_buffers_, out_device_buffer_);
            if(op_ptr->IsSupportedArgument(argument.get()))
            {
                std::string op_name = op_ptr->GetTypeString();
                float avg_time = invoker->Run(argument.get(), StreamConfig{nullptr, time_kernel});

                std::size_t flops     = op_instance_.GetFlops();
                std::size_t num_btype = op_instance_.GetBtype();
                float tflops          = static_cast<float>(flops) / 1.E9 / avg_time;
                float gb_per_sec      = num_btype / 1.E6 / avg_time;

                std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                          << " GB/s, " << op_name << std::endl;

                if(avg_time < best_config.best_avg_time)
                {
                    best_config.best_op_name    = op_name;
                    best_config.best_tflops     = tflops;
                    best_config.best_gb_per_sec = gb_per_sec;
                    best_config.best_avg_time   = avg_time;
                }

                if(do_verification)
                {
                    out_device_buffer_->FromDevice(out_tensor_->mData.data());
                    if(!ref_output_)
                    {
                        throw std::runtime_error(
                            "OpInstanceRunEngine::Profile: Reference value not availabe."
                            " You have to provide reference function.");
                    }
                    // TODO: enable flexible use of custom check_error functions
                    CheckErr(out_tensor_->mData, ref_output_->mData);

                    if(do_log) {}
                }
                out_device_buffer_->SetZero();
            }
        }
        return best_config;
    }

    void SetAtol(double a) { atol_ = a; }
    void SetRtol(double r) { rtol_ = r; }

    private:
    template <typename F, std::size_t... Is>
    void CallRefOpUnpackArgs(const F& f, std::index_sequence<Is...>) const
    {
        f(*std::get<Is>(in_tensors_)..., *ref_output_);
    }

    template <std::size_t... Is>
    void AllocateDeviceInputTensors(std::index_sequence<Is...>)
    {
        (AllocateDeviceInputTensorsImpl<Is>(), ...);
    }

    template <std::size_t Index>
    void AllocateDeviceInputTensorsImpl()
    {
        const auto& ts = std::get<Index>(in_tensors_);
        in_device_buffers_
            .emplace_back(
                std::make_unique<DeviceMem>(sizeof(std::tuple_element_t<Index, InArgsTypesTuple>) *
                                            ts->mDesc.GetElementSpaceSize()))
            ->ToDevice(ts->mData.data());
    }

    static constexpr std::size_t kNInArgs_ = std::tuple_size_v<InTensorsTuple>;
    const OpInstanceT& op_instance_;
    double rtol_{1e-5};
    double atol_{1e-8};

    InTensorsTuple in_tensors_;
    TensorPtr<OutDataType> out_tensor_;
    TensorPtr<OutDataType> ref_output_;

    DeviceBuffers in_device_buffers_;
    DeviceMemPtr out_device_buffer_;

    template <typename T>
    bool CheckErr(const std::vector<T>& dev_out, const std::vector<T>& ref_out) const
    {
        return ck::utils::check_err(dev_out, ref_out, "Error: incorrect results!", rtol_, atol_);
    }
};

} // namespace utils
} // namespace ck
