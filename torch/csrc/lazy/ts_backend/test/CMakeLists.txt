# This points back to the pytorch/lazy_tensor_core tree that we are moving away from
# it should be removed once migration is complete
# I left some stuff there rather than moving it all, to indicate things we may not
# ever merge, and should consider redesigining or replace utils with ATen/torch native ones
set(PYTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")
set(OLD_LTC_DIR "${PYTORCH_ROOT}/lazy_tensor_core")
set(LTC_DIR "${PYTORCH_ROOT}/torch/csrc/lazy")

file(GLOB TEST_LTC_TS_SRC
     "*.cpp"
)
add_executable(test_ltc_ts ${TEST_LTC_TS_SRC})
target_link_libraries(test_ltc_ts PRIVATE gtest torch torch_ltc_ts old_ltc_util)

# DISABLE Werror for XLA deps
# TODO(whc) should remove these after cleaning up utils/deps from XLA
SET_TARGET_PROPERTIES(test_ltc_ts PROPERTIES 
                      COMPILE_FLAGS
                      -Wno-error=return-type )
