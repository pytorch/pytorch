# This points back to the pytorch/lazy_tensor_core tree that we are moving away from
# it should be removed once migration is complete
# I left some stuff there rather than moving it all, to indicate things we may not
# ever merge, and should consider redesigining or replace utils with ATen/torch native ones
set(PYTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
set(OLD_LTC_DIR "${PYTORCH_ROOT}/lazy_tensor_core")

# We generally want to move off these dependencies and switch to native 
# equivalents already inside pytorch, or, move things from here into
# either pytorch or lazy/core case-by-case.  End state this lib should
# disappear.
file(GLOB OLD_LTC_UTIL_SRC 
     ${OLD_LTC_DIR}/third_party/lazy_tensors/computation_client/*.cc
     ${OLD_LTC_DIR}/lazy_tensors/*.cc
     ${OLD_LTC_DIR}/lazy_tensor_core/csrc/*.cc
     ${OLD_LTC_DIR}/lazy_tensor_core/csrc/*.cpp

     ${OLD_LTC_DIR}/lazy_tensors/core/platform/*.cc)
add_library(old_ltc_util ${OLD_LTC_UTIL_SRC})
target_include_directories(old_ltc_util
                           PUBLIC
                           "${OLD_LTC_DIR}"
                           "${OLD_LTC_DIR}/third_party")
SET_TARGET_PROPERTIES(old_ltc_util PROPERTIES
                      COMPILE_FLAGS
                      -Wno-error=return-type )


file(GLOB LTC_SRC
     "lazy_tensor_core/csrc/*.cpp"
     # TODO(whc)
     # ops should be part of ts_backend, but, stuff in core currently depends on them
     "lazy_tensor_core/csrc/ops/*.cpp")
add_library(torch_ltc ${LTC_SRC})
target_link_libraries(torch_ltc PUBLIC old_ltc_util)
target_include_directories(torch_ltc
                           PUBLIC "${PYTORCH_ROOT}/torch/csrc/lazy/core")

# TODO(whc)
# something in here depends on python. I don't know what it is yet, but it should probably
# be moved to the TS backend (or removed).  Adding it for now to get things building with 
# minimal changes.
target_link_libraries(torch_ltc PUBLIC ${PYTHON_LIBRARIES})

# DISABLE Werror for XLA deps
# TODO(whc) should remove these after cleaning up utils/deps from XLA
SET_TARGET_PROPERTIES(torch_ltc PROPERTIES
                      COMPILE_FLAGS
                      -Wno-error=return-type )
