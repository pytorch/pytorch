// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

int run_gemm_test()
{
    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    auto test = [&](auto a_layout, auto b_layout, auto c_layout) {
        bool pass = true;

        using DeviceOp = ck::tensor_operation::device::DeviceGemm<decltype(a_layout),
                                                                  decltype(b_layout),
                                                                  decltype(c_layout),
                                                                  ADataType,
                                                                  BDataType,
                                                                  CDataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>;

        const auto gemmPtrs =
            ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
                DeviceOp>::GetInstances();

        for(auto& gemmPtr : gemmPtrs)
        {
            pass &= ck::gemm_util::TestGemm<AccDataType>{}(gemmPtr.get());
        }

        return pass;
    };

    bool pass = test(Row{}, Row{}, Row{}) && test(Row{}, Col{}, Row{}) &&
                test(Col{}, Row{}, Row{}) && test(Col{}, Col{}, Row{});

    std::cout << "TestGemm ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
