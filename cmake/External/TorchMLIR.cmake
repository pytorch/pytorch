if(NOT USE_MLIR_EXPORTER)
    message(WARNING "NOT Enabling Torch MLIR Exporter")
    return()
endif()

message(WARNING "Enabling Torch MLIR Exporter")

set(MLIR_EXPORTER_SOURCE https://github.com/llvm/torch-mlir)
set(MLIR_BUILD ${CMAKE_BINARY_DIR}/torch-mlir)
set(MLIR_INSTALL ${CMAKE_BINARY_DIR}/torch-mlir-install)

ExternalProject_Add(torch-mlir
    GIT_REPOSITORY ${MLIR_EXPORTER_SOURCE}
    GIT_SUBMODULES_RECURSE ON
    GIT_TAG main
    GIT_PROGRESS ON
    PREFIX ${MLIR_BUILD}
    BINARY_DIR ${MLIR_BUILD}
    INSTALL_DIR ${MLIR_INSTALL}
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -GNinja -DCMAKE_INSTALL_PREFIX=${MLIR_INSTALL} -DLLVM_ENABLE_PROJECTS=mlir
      -DLLVM_EXTERNAL_PROJECTS=torch-mlir -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=${MLIR_BUILD}/src/torch-mlir
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DLLVM_TARGETS_TO_BUILD=host
      ${MLIR_BUILD}/src/torch-mlir/external/llvm-project/llvm
)
