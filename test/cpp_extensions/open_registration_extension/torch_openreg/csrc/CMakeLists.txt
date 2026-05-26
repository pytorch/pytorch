set(LIBRARY_NAME torch_openreg)

# Generate the PrivateUse1 dispatch registrations and native-function
# declarations from the backend YAML using torchgen (gen_backend_stubs). This
# exercises the structured-kernel + out-as-primary codegen path end to end: the
# generated RegisterPrivateUse1.cpp is compiled into the backend library, and the
# generated PrivateUse1NativeFunctions.h is consumed by aten/OpenRegStructured.cpp.
set(OPENREG_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(OPENREG_BACKEND_YAML
    "${CMAKE_CURRENT_SOURCE_DIR}/aten/openreg_native_functions.yaml")
file(MAKE_DIRECTORY ${OPENREG_GEN_DIR})
add_custom_command(
    OUTPUT
        "${OPENREG_GEN_DIR}/RegisterPrivateUse1.cpp"
        "${OPENREG_GEN_DIR}/PrivateUse1NativeFunctions.h"
    COMMAND ${PYTHON_EXECUTABLE} -m torchgen.gen_backend_stubs
        --source-yaml "${OPENREG_BACKEND_YAML}"
        --output-dir "${OPENREG_GEN_DIR}"
    DEPENDS "${OPENREG_BACKEND_YAML}"
    COMMENT "torchgen: generating PrivateUse1 stubs for openreg"
    VERBATIM
)
add_custom_target(openreg_codegen
    DEPENDS
        "${OPENREG_GEN_DIR}/RegisterPrivateUse1.cpp"
        "${OPENREG_GEN_DIR}/PrivateUse1NativeFunctions.h"
)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
list(APPEND SOURCE_FILES "${OPENREG_GEN_DIR}/RegisterPrivateUse1.cpp")

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})
add_dependencies(${LIBRARY_NAME} openreg_codegen)

target_link_libraries(${LIBRARY_NAME} PRIVATE torch_cpu_library openreg)
target_include_directories(${LIBRARY_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OPENREG_GEN_DIR}
)

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
