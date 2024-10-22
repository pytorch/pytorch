#include <gtest/gtest.h>
#include <torch/torch.h>

#include <c10/core/impl/COWDeleter.h>
#include <ATen/mps/MPSCOWContext.h>

namespace at::mps {

namespace {

at::DataPtr& GetMutableDataPtr(torch::Tensor& t) {
    auto* storage_impl = t.storage().unsafeGetStorageImpl();
    return storage_impl->_mutable_data_ptr_no_checks();
}

}  // namespace

TEST(MPSCOWContext, MPSToCPU) {
    torch::Tensor mps_t = torch::randn({10000}, at::device(at::kMPS));
    torch::Tensor cpu_t = at::_lazy_clone(mps_t, at::kCPU);

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    // They will share the same cow context.
    ASSERT_EQ(mps_dp.get_context(), cpu_dp.get_context());
    ASSERT_EQ(mps_dp.get_deleter(), c10::impl::cow::cow_deleter);
    ASSERT_EQ(cpu_dp.get_deleter(), c10::impl::cow::cow_deleter);

    auto* ctx = mps_dp.cast_context<c10::impl::cow::COWDeleterContext>(
        c10::impl::cow::cow_deleter
    );

    ASSERT_EQ(ctx->GetDataDeleter(), c10::impl::cow::unified_memory_data_ptr_ctx_deleter);

    const auto* unimem_ctx =
        reinterpret_cast<const at::mps::cow::MPSToCPUDataPtrContext*>(ctx->GetConstDataPtr());
    ASSERT_EQ(mps_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(cpu_dp.get(), unimem_ctx->get_mapped_data_ctx());
    ASSERT_FALSE(unimem_ctx->memory_backed_by_cpu());
}

TEST(MPSCOWContext, MPSToCPUMaterialization1) {
    torch::Tensor mps_t = torch::randn({10000}, at::device(at::kMPS));
    torch::Tensor cpu_t = at::_lazy_clone(mps_t, at::kCPU);
    mps_t += 1;
    cpu_t += 1;
    // In this case both MPS and CPU will get its own dedicated memory.

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    ASSERT_NE(mps_dp.get_context(), cpu_dp.get_context());
    ASSERT_NE(mps_dp.get_deleter(), cpu_dp.get_deleter());
}

TEST(MPSCOWContext, MPSToCPUMaterialization2) {
    torch::Tensor mps_t = torch::randn({10000}, at::device(at::kMPS));
    void* original_mps_pointer = GetMutableDataPtr(mps_t).get_context();
    torch::Tensor cpu_t = at::_lazy_clone(mps_t, at::kCPU);
    cpu_t += 1;
    mps_t += 1;
    // In this case MPS will own the original memory, CPU will get its
    // own dedicated memory.

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    ASSERT_EQ(mps_dp.get(), original_mps_pointer);
    ASSERT_EQ(mps_dp.get_context(), original_mps_pointer);

    ASSERT_NE(mps_dp.get_context(), cpu_dp.get_context());
    ASSERT_NE(mps_dp.get_deleter(), cpu_dp.get_deleter());
}

TEST(MPSCOWContext, CPUToMPS) {
    torch::Tensor cpu_t = torch::randn({10000}, at::device(at::kCPU));
    torch::Tensor mps_t = at::_lazy_clone(cpu_t, at::kMPS);

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    // They will share the same cow context.
    ASSERT_EQ(mps_dp.get_context(), cpu_dp.get_context());
    ASSERT_EQ(mps_dp.get_deleter(), c10::impl::cow::cow_deleter);
    ASSERT_EQ(cpu_dp.get_deleter(), c10::impl::cow::cow_deleter);

    auto* ctx = mps_dp.cast_context<c10::impl::cow::COWDeleterContext>(
        c10::impl::cow::cow_deleter
    );

    ASSERT_EQ(ctx->GetDataDeleter(), c10::impl::cow::unified_memory_data_ptr_ctx_deleter);

    const auto* unimem_ctx =
        reinterpret_cast<const at::mps::cow::CPUToMPSDataPtrContext*>(ctx->GetConstDataPtr());
    ASSERT_EQ(cpu_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(mps_dp.get(), unimem_ctx->get_mapped_data_ctx());
    ASSERT_TRUE(unimem_ctx->memory_backed_by_cpu());
}

TEST(MPSCOWContext, CPUToMPSMaterialization1) {
    torch::Tensor cpu_t = torch::randn({10000}, at::device(at::kCPU));
    torch::Tensor mps_t = at::_lazy_clone(cpu_t, at::kMPS);
    cpu_t += 1;
    mps_t += 1;
    // In this case both MPS and CPU will get its own dedicated memory.

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    ASSERT_NE(mps_dp.get_context(), cpu_dp.get_context());
    ASSERT_NE(mps_dp.get_deleter(), cpu_dp.get_deleter());
}

TEST(MPSCOWContext, CPUToMPSMaterialization2) {
    torch::Tensor cpu_t = torch::randn({10000}, at::device(at::kCPU));
    void* original_cpu_pointer = GetMutableDataPtr(cpu_t).get_context();
    torch::Tensor mps_t = at::_lazy_clone(cpu_t, at::kMPS);
    mps_t += 1;
    cpu_t += 1;
    // In this case CPU will own the original memory, MPS will get its
    // own dedicated memory.

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    ASSERT_EQ(cpu_dp.get(), original_cpu_pointer);
    ASSERT_EQ(cpu_dp.get_context(), original_cpu_pointer);

    ASSERT_NE(mps_dp.get_context(), cpu_dp.get_context());
    ASSERT_NE(mps_dp.get_deleter(), cpu_dp.get_deleter());
}

TEST(MPSCOWContext, MPSToCPUToMPS) {
    torch::Tensor mps_t = torch::randn({10000}, at::device(at::kMPS));
    torch::Tensor cpu_t = at::_lazy_clone(mps_t, at::kCPU);
    torch::Tensor mps2_t = at::_lazy_clone(mps_t);
    torch::Tensor mps3_t = at::_lazy_clone(cpu_t, at::kMPS);

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& mps2_dp = GetMutableDataPtr(mps2_t);
    auto& mps3_dp = GetMutableDataPtr(mps3_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    auto* ctx = mps_dp.cast_context<c10::impl::cow::COWDeleterContext>(
        c10::impl::cow::cow_deleter
    );

    ASSERT_EQ(ctx->GetDataDeleter(), c10::impl::cow::unified_memory_data_ptr_ctx_deleter);

    const auto* unimem_ctx =
        reinterpret_cast<const at::mps::cow::MPSToCPUDataPtrContext*>(ctx->GetConstDataPtr());
    ASSERT_EQ(mps_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(mps2_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(mps3_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(cpu_dp.get(), unimem_ctx->get_mapped_data_ctx());
    ASSERT_FALSE(unimem_ctx->memory_backed_by_cpu());
}

TEST(MPSCOWContext, CPUToMPSToCPU) {
    torch::Tensor cpu_t = torch::randn({10000}, at::device(at::kCPU));
    torch::Tensor mps_t = at::_lazy_clone(cpu_t, at::kMPS);
    torch::Tensor cpu2_t = at::_lazy_clone(cpu_t);
    torch::Tensor cpu3_t = at::_lazy_clone(mps_t, at::kCPU);

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& cpu2_dp = GetMutableDataPtr(cpu2_t);
    auto& cpu3_dp = GetMutableDataPtr(cpu3_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    auto* ctx = cpu_dp.cast_context<c10::impl::cow::COWDeleterContext>(
        c10::impl::cow::cow_deleter
    );

    ASSERT_EQ(ctx->GetDataDeleter(), c10::impl::cow::unified_memory_data_ptr_ctx_deleter);

    const auto* unimem_ctx =
        reinterpret_cast<const at::mps::cow::CPUToMPSDataPtrContext*>(ctx->GetConstDataPtr());
    ASSERT_EQ(cpu2_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(cpu3_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(cpu_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(mps_dp.get(), unimem_ctx->get_mapped_data_ctx());
    ASSERT_TRUE(unimem_ctx->memory_backed_by_cpu());
}

// Test the `to` interface.

TEST(MPSCOWContextToImpl, MPSToCPU) {
    torch::Tensor mps_t = torch::randn({10000}, at::device(at::kMPS));
    torch::Tensor cpu_t = at::native::to(
        mps_t,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        /*device=*/at::kCPU,
        /*pin_memory=*/std::nullopt,
        false,
        false,
        /*optional_memory_format=*/std::nullopt
    );

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    // They will share the same cow context.
    ASSERT_EQ(mps_dp.get_context(), cpu_dp.get_context());
    ASSERT_EQ(mps_dp.get_deleter(), c10::impl::cow::cow_deleter);
    ASSERT_EQ(cpu_dp.get_deleter(), c10::impl::cow::cow_deleter);

    auto* ctx = mps_dp.cast_context<c10::impl::cow::COWDeleterContext>(
        c10::impl::cow::cow_deleter
    );

    ASSERT_EQ(ctx->GetDataDeleter(), c10::impl::cow::unified_memory_data_ptr_ctx_deleter);

    const auto* unimem_ctx =
        reinterpret_cast<const at::mps::cow::MPSToCPUDataPtrContext*>(ctx->GetConstDataPtr());
    ASSERT_EQ(mps_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(cpu_dp.get(), unimem_ctx->get_mapped_data_ctx());
    ASSERT_FALSE(unimem_ctx->memory_backed_by_cpu());
}

TEST(MPSCOWContextToImpl, MPSToCPUMaterialization1) {
    torch::Tensor mps_t = torch::randn({10000}, at::device(at::kMPS));
    torch::Tensor cpu_t = at::native::to(
        mps_t,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        /*device=*/at::kCPU,
        /*pin_memory=*/std::nullopt,
        false,
        false,
        /*optional_memory_format=*/std::nullopt
    );
    mps_t += 1;
    cpu_t += 1;
    // In this case both MPS and CPU will get its own dedicated memory.

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    ASSERT_NE(mps_dp.get_context(), cpu_dp.get_context());
    ASSERT_NE(mps_dp.get_deleter(), cpu_dp.get_deleter());
}

TEST(MPSCOWContextToImpl, MPSToCPUMaterialization2) {
    torch::Tensor mps_t = torch::randn({10000}, at::device(at::kMPS));
    void* original_mps_pointer = GetMutableDataPtr(mps_t).get_context();
    torch::Tensor cpu_t = at::native::to(
        mps_t,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        /*device=*/at::kCPU,
        /*pin_memory=*/std::nullopt,
        false,
        false,
        /*optional_memory_format=*/std::nullopt
    );
    cpu_t += 1;
    mps_t += 1;
    // In this case MPS will own the original memory, CPU will get its
    // own dedicated memory.

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    ASSERT_EQ(mps_dp.get(), original_mps_pointer);
    ASSERT_EQ(mps_dp.get_context(), original_mps_pointer);

    ASSERT_NE(mps_dp.get_context(), cpu_dp.get_context());
    ASSERT_NE(mps_dp.get_deleter(), cpu_dp.get_deleter());
}

TEST(MPSCOWContextToImpl, CPUToMPS) {
    torch::Tensor cpu_t = torch::randn({10000}, at::device(at::kCPU));
    torch::Tensor mps_t = at::native::to(
        cpu_t,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        /*device=*/at::kMPS,
        /*pin_memory=*/std::nullopt,
        false,
        false,
        /*optional_memory_format=*/std::nullopt
    );

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    // They will share the same cow context.
    ASSERT_EQ(mps_dp.get_context(), cpu_dp.get_context());
    ASSERT_EQ(mps_dp.get_deleter(), c10::impl::cow::cow_deleter);
    ASSERT_EQ(cpu_dp.get_deleter(), c10::impl::cow::cow_deleter);

    auto* ctx = mps_dp.cast_context<c10::impl::cow::COWDeleterContext>(
        c10::impl::cow::cow_deleter
    );

    ASSERT_EQ(ctx->GetDataDeleter(), c10::impl::cow::unified_memory_data_ptr_ctx_deleter);

    const auto* unimem_ctx =
        reinterpret_cast<const at::mps::cow::CPUToMPSDataPtrContext*>(ctx->GetConstDataPtr());
    ASSERT_EQ(cpu_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(mps_dp.get(), unimem_ctx->get_mapped_data_ctx());
    ASSERT_TRUE(unimem_ctx->memory_backed_by_cpu());
}

TEST(MPSCOWContextToImpl, CPUToMPSMaterialization1) {
    torch::Tensor cpu_t = torch::randn({10000}, at::device(at::kCPU));
    torch::Tensor mps_t = at::native::to(
        cpu_t,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        /*device=*/at::kMPS,
        /*pin_memory=*/std::nullopt,
        false,
        false,
        /*optional_memory_format=*/std::nullopt
    );
    cpu_t += 1;
    mps_t += 1;
    // In this case both MPS and CPU will get its own dedicated memory.

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    ASSERT_NE(mps_dp.get_context(), cpu_dp.get_context());
    ASSERT_NE(mps_dp.get_deleter(), cpu_dp.get_deleter());
}

TEST(MPSCOWContextToImpl, CPUToMPSMaterialization2) {
    torch::Tensor cpu_t = torch::randn({10000}, at::device(at::kCPU));
    void* original_cpu_pointer = GetMutableDataPtr(cpu_t).get_context();
    torch::Tensor mps_t = at::native::to(
        cpu_t,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        /*device=*/at::kMPS,
        /*pin_memory=*/std::nullopt,
        false,
        false,
        /*optional_memory_format=*/std::nullopt
    );
    mps_t += 1;
    cpu_t += 1;
    // In this case CPU will own the original memory, MPS will get its
    // own dedicated memory.

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    ASSERT_EQ(cpu_dp.get(), original_cpu_pointer);
    ASSERT_EQ(cpu_dp.get_context(), original_cpu_pointer);

    ASSERT_NE(mps_dp.get_context(), cpu_dp.get_context());
    ASSERT_NE(mps_dp.get_deleter(), cpu_dp.get_deleter());
}

TEST(MPSCOWContextToImpl, MPSToCPUToMPS) {
    torch::Tensor mps_t = torch::randn({10000}, at::device(at::kMPS));
    torch::Tensor cpu_t = at::native::to(
        mps_t,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        /*device=*/at::kCPU,
        /*pin_memory=*/std::nullopt,
        false,
        false,
        /*optional_memory_format=*/std::nullopt
    );
    // This is an alias.
    torch::Tensor mps2_t = at::native::to(
        mps_t,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        /*device=*/std::nullopt,
        /*pin_memory=*/std::nullopt,
        false,
        false,
        /*optional_memory_format=*/std::nullopt
    );
    torch::Tensor mps3_t = at::native::to(
        cpu_t,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        /*device=*/at::kMPS,
        /*pin_memory=*/std::nullopt,
        false,
        false,
        /*optional_memory_format=*/std::nullopt
    );;

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& mps2_dp = GetMutableDataPtr(mps2_t);
    auto& mps3_dp = GetMutableDataPtr(mps3_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    auto* ctx = mps_dp.cast_context<c10::impl::cow::COWDeleterContext>(
        c10::impl::cow::cow_deleter
    );

    ASSERT_EQ(ctx->GetDataDeleter(), c10::impl::cow::unified_memory_data_ptr_ctx_deleter);

    const auto* unimem_ctx =
        reinterpret_cast<const at::mps::cow::MPSToCPUDataPtrContext*>(ctx->GetConstDataPtr());
    ASSERT_EQ(mps_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(mps2_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(mps3_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(cpu_dp.get(), unimem_ctx->get_mapped_data_ctx());
    ASSERT_FALSE(unimem_ctx->memory_backed_by_cpu());
}

TEST(MPSCOWContextToImpl, CPUToMPSToCPU) {
    torch::Tensor cpu_t = torch::randn({10000}, at::device(at::kCPU));
    torch::Tensor mps_t = at::native::to(
        cpu_t,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        /*device=*/at::kMPS,
        /*pin_memory=*/std::nullopt,
        false,
        false,
        /*optional_memory_format=*/std::nullopt
    );
    // This is an alias.
    torch::Tensor cpu2_t = at::native::to(
        cpu_t,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        /*device=*/std::nullopt,
        /*pin_memory=*/std::nullopt,
        false,
        false,
        /*optional_memory_format=*/std::nullopt
    );
    torch::Tensor cpu3_t = at::native::to(
        mps_t,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        /*device=*/at::kCPU,
        /*pin_memory=*/std::nullopt,
        false,
        false,
        /*optional_memory_format=*/std::nullopt
    );;

    auto& mps_dp = GetMutableDataPtr(mps_t);
    auto& cpu2_dp = GetMutableDataPtr(cpu2_t);
    auto& cpu3_dp = GetMutableDataPtr(cpu3_t);
    auto& cpu_dp = GetMutableDataPtr(cpu_t);

    auto* ctx = cpu_dp.cast_context<c10::impl::cow::COWDeleterContext>(
        c10::impl::cow::cow_deleter
    );

    ASSERT_EQ(ctx->GetDataDeleter(), c10::impl::cow::unified_memory_data_ptr_ctx_deleter);

    const auto* unimem_ctx =
        reinterpret_cast<const at::mps::cow::CPUToMPSDataPtrContext*>(ctx->GetConstDataPtr());
    ASSERT_EQ(cpu2_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(cpu3_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(cpu_dp.get(), unimem_ctx->get_original_data_ctx());
    ASSERT_EQ(mps_dp.get(), unimem_ctx->get_mapped_data_ctx());
    ASSERT_TRUE(unimem_ctx->memory_backed_by_cpu());
}

}  // namespace at::mps
