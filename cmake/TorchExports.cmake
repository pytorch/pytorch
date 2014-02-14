INSTALL(EXPORT torch-exports
  DESTINATION "${Torch_INSTALL_CMAKE_SUBDIR}"
  FILE "TorchExports.cmake")

CONFIGURE_FILE("cmake/TorchConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/TorchConfig.cmake" @ONLY)
CONFIGURE_FILE("cmake/TorchWrap.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/TorchWrap.cmake" @ONLY)

INSTALL(
  FILES
  "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/TorchConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/TorchWrap.cmake"
  "cmake/TorchPathsInit.cmake"
  "cmake/TorchPackage.cmake"
  DESTINATION "${Torch_INSTALL_CMAKE_SUBDIR}")
