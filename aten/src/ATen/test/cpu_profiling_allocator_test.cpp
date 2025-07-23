#include <gtest/gtest.h>

#include <c10/core/CPUAllocator.h>
#include <c10/mobile/CPUProfilingAllocator.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>

at::Tensor run_with_control_flow(
    at::Tensor input,
    at::Tensor conv_weight,
    at::Tensor linear_weight,
    bool cond,
    std::vector<void*>& pointers,
    bool record = false,
    bool validate = false) {
  if (cond) {
    input = input * 2;
  }
  void* input_ptr = input.data_ptr();
  auto conv_out = at::conv2d(input, conv_weight);
  void* conv_out_ptr = input.data_ptr();
  auto conv_out_flat = conv_out.view({conv_out.size(0), -1});
  auto output = at::linear(conv_out_flat, linear_weight);
  if (record) {
    pointers.push_back(input_ptr);
    pointers.push_back(conv_out_ptr);
  }
  if (validate) {
    TORCH_CHECK(input_ptr == pointers[0]);
    TORCH_CHECK(conv_out_ptr == pointers[1]);
  }
  return output;
}

TEST(CPUAllocationPlanTest, with_control_flow) {
  at::Tensor a = at::rand({23, 16, 16, 16});
  at::Tensor conv_weight = at::rand({16, 16, 3, 3});
  // output shape
  // 23, 16, 14, 14
  // Flattened shape = 23, 3136
  at::Tensor linear_weight = at::rand({32, 3136});
  at::Tensor output, ref_output;
  std::vector<void*> pointers;

  auto valid_allocation_plan = [&]() {
    c10::AllocationPlan plan;
    {
      c10::WithProfileAllocationsGuard profile_guard(&plan);
      ref_output = run_with_control_flow(
          a, conv_weight, linear_weight, true, pointers);
    }
  };
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_NO_THROW(valid_allocation_plan());

  auto validate_allocation_plan =
    [&](bool record_mode, bool validation_mode) -> bool {
    c10::AllocationPlan plan;
    {
      c10::WithProfileAllocationsGuard profile_guard(&plan);
      ref_output =
        run_with_control_flow(a, conv_weight, linear_weight, record_mode, pointers);
    }
    bool success{true};
    for (uint64_t i = 0; i < 10; ++i) {
      bool validation_success = false;
      {
        c10::WithValidateAllocationPlanGuard
          validation_guard(&plan, &validation_success);
        output = run_with_control_flow(
            a, conv_weight, linear_weight, validation_mode, pointers);
      }
      success = success && validation_success;
    }
    return success;
  };
  ASSERT_TRUE(validate_allocation_plan(true, true));
  ASSERT_TRUE(validate_allocation_plan(false, false));

  #ifdef C10_MOBILE
  // Returning false when record mode != validation mode only applies to
  // DefaultMobileCPUAllocator, DefaultCPUAllocator has no such behavior
  // and will always return true
  ASSERT_FALSE(validate_allocation_plan(false, true));
  ASSERT_FALSE(validate_allocation_plan(true, false));
  #else
  ASSERT_TRUE(validate_allocation_plan(false, true));
  ASSERT_TRUE(validate_allocation_plan(true, false));
  #endif
}

TEST(CPUAllocationPlanTest, with_profiling_alloc) {
  at::Tensor a = at::rand({23, 16, 16, 16});
  at::Tensor conv_weight = at::rand({16, 16, 3, 3});
  // output shape
  // 23, 16, 14, 14
  // Flattened shape = 23, 3136
  at::Tensor linear_weight = at::rand({32, 3136});
  at::Tensor output, ref_output;
  std::vector<void*> pointers;

  auto valid_allocation_plan = [&]() {
    c10::AllocationPlan plan;
    {
      c10::WithProfileAllocationsGuard profile_guard(&plan);
      ref_output = run_with_control_flow(
          a, conv_weight, linear_weight, false, pointers);
    }
  };
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_NO_THROW(valid_allocation_plan());

  auto validate_allocation_plan =
    [&](bool record_mode,
        bool validation_mode,
        bool validate_pointers) {
      pointers.clear();
      c10::AllocationPlan plan;
      {
        c10::WithProfileAllocationsGuard profile_guard(&plan);
        ref_output = run_with_control_flow(
            a,
            conv_weight,
            linear_weight,
            record_mode,
            pointers,
            false,
            false);
      }
      c10::CPUProfilingAllocator profiling_allocator;
      {
        c10::WithProfilingAllocatorGuard
          profiling_allocator_guard(&profiling_allocator, &plan);
        output = run_with_control_flow(
            a,
            conv_weight,
            linear_weight,
            validation_mode,
            pointers,
            validate_pointers,
            false);
      }
      for (uint64_t i = 0; i < 10; ++i) {
        {
          c10::WithProfilingAllocatorGuard
            profiling_allocator_guard(&profiling_allocator, &plan);
          output = run_with_control_flow(
              a,
              conv_weight,
              linear_weight,
              validation_mode,
              pointers,
              false,
              validate_pointers);
        }
      }
  };
  // When control flow conditions are same between profiling and evaluation
  // profiling allocator should not throw.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_NO_THROW(validate_allocation_plan(true, true, false));
  ASSERT_TRUE(ref_output.equal(output));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_NO_THROW(validate_allocation_plan(false, false, false));
  ASSERT_TRUE(ref_output.equal(output));

  // Returning the same pointers is not a guarantee when the default
  // allocator is used. It looks like the underlying memory pointer
  // can change as long as output and ref_output remain equal. This
  // has already been confirmed in the previous two tests
  #ifdef C10_MOBILE
  // Furthermore profiling allocator should return the same pointers
  // back for the intermediate tensors
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_NO_THROW(validate_allocation_plan(true, true, true));
  ASSERT_TRUE(ref_output.equal(output));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_NO_THROW(validate_allocation_plan(false, false, true));
  ASSERT_TRUE(ref_output.equal(output));

  // When control flow conditions are different between profiling and evaluation
  // profiling allocator should throw.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(validate_allocation_plan(true, false, false), c10::Error);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(validate_allocation_plan(false, true, false), c10::Error);
  #else
  // Throwing when record mode != validation mode only applies to
  // DefaultMobileCPUAllocator, DefaultCPUAllocator has no such
  // behavior and throw nothing
  ASSERT_NO_THROW(validate_allocation_plan(true, false, false));
  ASSERT_NO_THROW(validate_allocation_plan(false, true, false));
  #endif
}

int main(int argc, char* argv[]) {
  // Setting the priority high to make sure no other allocator gets used instead of this.
  c10::SetCPUAllocator(c10::GetDefaultCPUAllocator(), /*priority*/ 100);

  #ifdef C10_MOBILE
  // Need to disable mkldnn for this test since it allocated memory
  // via raw_allocate interface which requires context pointer and raw
  // pointer to be the same. Tis is not true for mobile allocator.
  at::globalContext().setUserEnabledMkldnn(false);
  #endif

  ::testing::InitGoogleTest(&argc, argv);
  at::manual_seed(42);
  return RUN_ALL_TESTS();
}
