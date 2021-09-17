# This points back to the pytorch/lazy_tensor_core tree that we are moving away from
# it should be removed once migration is complete
# I left some stuff there rather than moving it all, to indicate things we may not
# ever merge, and should consider redesigining or replace utils with ATen/torch native ones
set(PYTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
set(OLD_LTC_DIR "${PYTORCH_ROOT}/lazy_tensor_core")
set(LTC_DIR "${PYTORCH_ROOT}/torch/csrc/lazy")

# Generate the native function stubs and dispatcher registrations
set(TS_BACKEND_GEN_SRC
    ${LTC_DIR}/ts_backend/lazy_tensor_core/csrc/ts_backend/LazyNativeFunctions.h 
    ${LTC_DIR}/ts_backend/lazy_tensor_core/csrc/ts_backend/RegisterAutogradLazy.cpp
    ${LTC_DIR}/ts_backend/lazy_tensor_core/csrc/ts_backend/RegisterLazy.cpp
)
# TODO(whc) does this properly 'depend' on the codegen sources?
add_custom_command(OUTPUT ${TS_BACKEND_GEN_SRC}
                COMMAND
                    "${PYTHON_EXECUTABLE}" -m tools.codegen.gen_lazy_tensor
                    --output_dir="${LTC_DIR}/ts_backend/lazy_tensor_core/csrc/ts_backend"
                    --source_yaml="${LTC_DIR}/ts_backend/ts_native_functions.yaml"
                    --impl_path="${LTC_DIR}/ts_backend/lazy_tensor_core/csrc/ts_backend/aten_ltc_ts_type.cpp"
                    --gen_ts_lowerings
                DEPENDS
                    ${LTC_DIR}/ts_backend/ts_native_functions.yaml)

add_custom_target(__ts_backend_gen DEPENDS ${TS_BACKEND_GEN_SRC})


file(GLOB LTC_TS_SRC
     ${TS_BACKEND_GEN_SRC}
     "lazy_tensor_core/csrc/ts_backend/*.cpp"
)
add_library(torch_ltc_ts ${LTC_TS_SRC})

# TODO(whc) this seemed to fix codegen not running, so, debug actual file-artifact dependency
add_dependencies(torch_ltc_ts __ts_backend_gen)

target_include_directories(torch_ltc_ts
                           PUBLIC
                           "${LTC_DIR}/ts_backend"
                           PRIVATE
                           # to get torch/jit.h
                           "${PYTORCH_ROOT}/torch/csrc/api/include"
                           "${OLD_LTC_DIR}"
                           "${OLD_LTC_DIR}/third_party")

target_include_directories(torch_ltc_ts SYSTEM
                           # TODO(whc) may be a cleaner way to handle the dep sequence below:
                           # init_python_bindings.cpp->python/pybind.h->THP.h->TH/THGeneral.h
                           PRIVATE "${PYTORCH_ROOT}/torch/include")
target_link_libraries(torch_ltc_ts PUBLIC torch_ltc)

# DISABLE Werror for XLA deps
# TODO(whc) should remove these after cleaning up utils/deps from XLA
SET_TARGET_PROPERTIES(torch_ltc_ts PROPERTIES 
                      COMPILE_FLAGS
                      -Wno-error=return-type )

add_subdirectory(test)