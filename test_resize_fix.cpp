/**
 * Test for the resize_ exception safety fix.
 * This test verifies that the fix for issue #170298 works correctly.
 * 
 * The fix ensures that tensor metadata is only updated AFTER storage resize
 * succeeds, preventing "zombie" tensors with inconsistent state.
 */

#include <gtest/gtest.h>
#include <ATen/ATen.h>
#include <ATen/native/Resize.h>
#include <c10/core/CPUAllocator.h>

namespace {

// Custom non-resizable storage implementation for testing
class NonResizableStorageImpl : public c10::StorageImpl {
public:
    NonResizableStorageImpl(size_t size_bytes)
        : StorageImpl(
            c10::StorageImpl::use_byte_size_t(),
            size_bytes,
            c10::GetCPUAllocator(),
            false  // resizable = false
          ) {}
    
    bool resizable() const override {
        return false;  // Always non-resizable
    }
};

class ResizeExceptionSafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a tensor with non-resizable storage
        auto storage = c10::make_intrusive<NonResizableStorageImpl>(0);  // 0 bytes
        tensor = at::empty({0}, at::dtype(at::kInt32));
        tensor.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(storage));
        
        // Verify initial state
        ASSERT_EQ(tensor.sizes(), at::IntArrayRef({0}));
        ASSERT_EQ(tensor.storage().nbytes(), 0);
        ASSERT_FALSE(tensor.storage().resizable());
    }
    
    at::Tensor tensor;
};

TEST_F(ResizeExceptionSafetyTest, ResizeFailsButPreservesOriginalShape) {
    // Record original state
    auto original_shape = tensor.sizes();
    auto original_storage_bytes = tensor.storage().nbytes();
    
    // Attempt to resize to a larger size (should fail due to non-resizable storage)
    EXPECT_THROW({
        tensor.resize_({5, 5, 5});
    }, std::runtime_error);
    
    // Verify that tensor metadata was NOT updated (exception safety)
    EXPECT_EQ(tensor.sizes(), original_shape);
    EXPECT_EQ(tensor.storage().nbytes(), original_storage_bytes);
    
    // Verify tensor is still in a consistent state (not "zombie")
    EXPECT_EQ(tensor.numel(), 0);  // Shape indicates 0 elements
    EXPECT_EQ(tensor.storage().nbytes(), 0);  // Storage has 0 bytes
    
    // Tensor should be safely accessible (no crash)
    EXPECT_NO_THROW({
        auto str = tensor.toString();  // Should not crash
        (void)str;  // Suppress unused variable warning
    });
}

TEST_F(ResizeExceptionSafetyTest, ResizeToSameSizeSucceeds) {
    // Resizing to the same size should succeed (no storage resize needed)
    EXPECT_NO_THROW({
        tensor.resize_({0});
    });
    
    // State should remain unchanged
    EXPECT_EQ(tensor.sizes(), at::IntArrayRef({0}));
    EXPECT_EQ(tensor.storage().nbytes(), 0);
}

TEST_F(ResizeExceptionSafetyTest, ResizeWithResizableStorageSucceeds) {
    // Create a tensor with resizable storage
    at::Tensor resizable_tensor = at::empty({2, 2}, at::dtype(at::kInt32));
    
    // Verify initial state
    EXPECT_EQ(resizable_tensor.sizes(), at::IntArrayRef({2, 2}));
    EXPECT_TRUE(resizable_tensor.storage().resizable());
    
    // Resize should succeed
    EXPECT_NO_THROW({
        resizable_tensor.resize_({3, 3});
    });
    
    // Verify new state
    EXPECT_EQ(resizable_tensor.sizes(), at::IntArrayRef({3, 3}));
    EXPECT_GE(resizable_tensor.storage().nbytes(), 3 * 3 * sizeof(int32_t));
}

} // namespace

// Test runner (if this file is compiled as a standalone test)
#ifndef GTEST_MAIN_DEFINED
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
