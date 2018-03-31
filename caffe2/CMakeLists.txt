# ---[ Declare source file lists

# ---[ Add respective subdirectories
# Note: the folders that are being commented out have not been properly
# addressed yet.

add_subdirectory(proto)

add_subdirectory(contrib)
add_subdirectory(core)
add_subdirectory(core/nomnigraph)
add_subdirectory(cuda_rtc)
add_subdirectory(db)
add_subdirectory(distributed)
# add_subdirectory(experiments) # note, we may remove this folder at some point
add_subdirectory(image)
add_subdirectory(video)
add_subdirectory(mkl)
add_subdirectory(mobile)
add_subdirectory(mpi)
add_subdirectory(observers)
add_subdirectory(onnx)
add_subdirectory(operators)
add_subdirectory(operators/rnn)
add_subdirectory(perfkernels)
add_subdirectory(python)
add_subdirectory(queue)
add_subdirectory(sgd)
add_subdirectory(share)
# add_subdirectory(test) # todo: use caffe2_gtest_main instead of gtest_main because we will need to call GlobalInit
add_subdirectory(transforms)
add_subdirectory(utils)

# Advanced: if we have white list specified, we will do intersections for all
# main lib srcs.
if (CAFFE2_WHITELISTED_FILES)
  caffe2_do_whitelist(Caffe2_CPU_SRCS CAFFE2_WHITELISTED_FILES)
  caffe2_do_whitelist(Caffe2_GPU_SRCS CAFFE2_WHITELISTED_FILES)
endif()

# Debug messages - if you want to get a list of source files, enable the
# following.
if (FALSE)
  message(STATUS "CPU sources: ")
  foreach(tmp ${Caffe2_CPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU sources: ")
  foreach(tmp ${Caffe2_GPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "CPU test sources: ")
  foreach(tmp ${Caffe2_CPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU test sources: ")
  foreach(tmp ${Caffe2_GPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()
endif()

# ---[ Generate and install header files.

# Write the macros file.
configure_file(
    ${PROJECT_SOURCE_DIR}/caffe2/core/macros.h.in
    ${PROJECT_BINARY_DIR}/caffe2/core/macros.h)

# Installing the header files
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
install(FILES ${PROJECT_BINARY_DIR}/caffe2/core/macros.h
        DESTINATION include/caffe2/core)


# ---[ List of libraries to link with
add_library(caffe2_protos STATIC $<TARGET_OBJECTS:Caffe_PROTO> $<TARGET_OBJECTS:Caffe2_PROTO>)
add_dependencies(caffe2_protos Caffe_PROTO Caffe2_PROTO)
# If we are going to link protobuf locally inside caffe2 libraries, what we will do is
# to create a helper static library that always contains libprotobuf source files, and
# link the caffe2 related dependent libraries to it.
target_include_directories(caffe2_protos INTERFACE $<INSTALL_INTERFACE:include>)
# Reason for this public dependency is as follows:
# (1) Strictly speaking, we should not expose any Protobuf related functions. We should
#     only use function interfaces wrapped with our own public API, and link protobuf
#     locally.
# (2) However, currently across the Caffe2 codebase, we have extensive use of protobuf
#     functionalities. For example, not only libcaffe2.so uses it, but also other
#     binaries such as python extensions etc. As a result, we will have to have a
#     transitive dependency to libprotobuf.
#
# Good thing is that, if we specify CAFFE2_LINK_LOCAL_PROTOBUF, then we do not need to
# separately deploy protobuf binaries - libcaffe2.so will contain all functionalities
# one needs. One can verify this via ldd.
#
# TODO item in the future includes:
# (1) Enable using lite protobuf
# (2) Properly define public API that do not directly depend on protobuf itself.
# (3) Expose the libprotobuf.a file for dependent libraries to link to.
#
# What it means for users/developers?
# (1) Users: nothing affecting the users, other than the fact that CAFFE2_LINK_LOCAL_PROTOBUF
#     avoids the need to deploy protobuf.
# (2) Developers: if one simply uses core caffe2 functionality without using protobuf,
#     nothing changes. If one has a dependent library that uses protobuf, then one needs to
#     have the right protobuf version as well as linking to libprotobuf.a.
target_link_libraries(caffe2_protos PUBLIC protobuf::libprotobuf)

# Compile exposed libraries.
add_library(caffe2 ${Caffe2_CPU_SRCS})
caffe2_interface_library(caffe2_protos caffe2_protos_whole)
target_link_libraries(caffe2 PRIVATE caffe2_protos_whole)
if (${CAFFE2_LINK_LOCAL_PROTOBUF})
  target_link_libraries(caffe2 INTERFACE protobuf::libprotobuf)
else()
  target_link_libraries(caffe2 PUBLIC protobuf::libprotobuf)
endif()
target_link_libraries(caffe2 PUBLIC ${Caffe2_PUBLIC_DEPENDENCY_LIBS})
target_link_libraries(caffe2 PRIVATE ${Caffe2_DEPENDENCY_LIBS})
target_link_libraries(caffe2 PRIVATE ${Caffe2_DEPENDENCY_WHOLE_LINK_LIBS})
target_include_directories(caffe2 INTERFACE $<INSTALL_INTERFACE:include>)
target_compile_options(caffe2 INTERFACE "-std=c++11")
target_compile_options(caffe2 PRIVATE "-DCAFFE2_BUILD_MAIN_LIB")
# Use -O2 for release builds (-O3 doesn't improve perf, and -Os results in perf regression)
target_compile_options(caffe2 PRIVATE "$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-O2>")
install(TARGETS caffe2 EXPORT Caffe2Targets DESTINATION lib)
caffe2_interface_library(caffe2 caffe2_library)
list(APPEND Caffe2_MAIN_LIBS caffe2_library)

# ---[ CUDA library.
if(USE_CUDA)
  # A hack to deal with cuda library dependencies and modern CMake: the
  # CUDA_ADD_LIBRARY includes a target_link_libraries, and as a result,
  # one cannot use PUBLIC/PRIVATE/INTERFACE for the target anymore. This
  # hack adds the PRIVATE keywords to CUDA_LIBRARIES so we can deal with
  # it. We will then manually add the cudart library as interface libs.
  set(__tmp ${CUDA_LIBRARIES})
  set(CUDA_LIBRARIES PRIVATE ${CUDA_LIBRARIES})
  CUDA_ADD_LIBRARY(caffe2_gpu ${Caffe2_GPU_SRCS})
  set(CUDA_LIBRARIES ${__tmp})
  target_link_libraries(caffe2_gpu INTERFACE caffe2::cudart)

  target_include_directories(
      caffe2_gpu INTERFACE $<INSTALL_INTERFACE:include>)
  target_link_libraries(
      caffe2_gpu PUBLIC caffe2 ${Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS})
  target_link_libraries(
      caffe2_gpu PRIVATE ${Caffe2_CUDA_DEPENDENCY_LIBS})
  caffe2_interface_library(caffe2_gpu caffe2_gpu_library)
  list(APPEND Caffe2_MAIN_LIBS caffe2_gpu_library)
  install(TARGETS caffe2_gpu EXPORT Caffe2Targets DESTINATION lib)
endif()

# ---[ Test binaries.
if (BUILD_TEST)
  set(Caffe2_ALL_TEST_SRCS ${Caffe2_CPU_TEST_SRCS})
  if (USE_CUDA)
    list(APPEND Caffe2_ALL_TEST_SRCS ${Caffe2_GPU_TEST_SRCS})
  endif()

  foreach(test_src ${Caffe2_ALL_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    # For tests, some of the test code actually directly call the dependent
    # libraries even if they are not part of the public dependency libs. As a
    # result, we will explicitly link the test against the Caffe2 dependency
    # libs.
    target_link_libraries(${test_name} ${Caffe2_MAIN_LIBS} gtest_main)
    if (USE_CUDA)
      target_link_libraries(${test_name} ${Caffe2_CUDA_DEPENDENCY_LIBS})
    endif()
    if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.0)
      target_compile_features(${test_name} PRIVATE cxx_range_for)
    endif()
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    install(TARGETS ${test_name} DESTINATION test)
  endforeach()
endif()


if (BUILD_PYTHON)
  # Python site-packages
  # Get canonical directory for python site packages (relative to install
  # location).  It varys from system to system.
  pycmd(PYTHON_SITE_PACKAGES "
      from distutils import sysconfig
      print(sysconfig.get_python_lib(prefix=''))
  ")
  # ---[ Options.
  SET(PYTHON_LIB_REL_PATH "${PYTHON_SITE_PACKAGES}" CACHE STRING "Python installation path (relative to CMake installation prefix)")
  message(STATUS "Using ${PYTHON_LIB_REL_PATH} as python relative installation path")
  # Python extension suffix
  # Try to get from python through sysconfig.get_env_var('EXT_SUFFIX') first,
  # fallback to ".pyd" if windows and ".so" for all others.
  pycmd(PY_EXT_SUFFIX "
      from distutils import sysconfig
      ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
      print(ext_suffix if ext_suffix else '')
  ")
  if("${PY_EXT_SUFFIX}" STREQUAL "")
    if (MSVC)
      set(PY_EXT_SUFFIX ".pyd")
    else()
      set(PY_EXT_SUFFIX ".so")
    endif()
  endif()

  # ---[ Python.
  add_library(caffe2_pybind11_state MODULE ${Caffe2_CPU_PYTHON_SRCS})
  set_target_properties(caffe2_pybind11_state PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  set_target_properties(caffe2_pybind11_state PROPERTIES PREFIX "")
  set_target_properties(caffe2_pybind11_state PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
  if (APPLE)
    set_target_properties(caffe2_pybind11_state PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif()
  set_target_properties(
      caffe2_pybind11_state PROPERTIES LIBRARY_OUTPUT_DIRECTORY
      ${CMAKE_BINARY_DIR}/caffe2/python)
  target_link_libraries(
      caffe2_pybind11_state caffe2_library)
  install(TARGETS caffe2_pybind11_state DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python")

  if(USE_CUDA)
    add_library(caffe2_pybind11_state_gpu MODULE ${Caffe2_GPU_PYTHON_SRCS})
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES PREFIX "")
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
    if (APPLE)
      set_target_properties(caffe2_pybind11_state_gpu PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    endif()
    set_target_properties(
        caffe2_pybind11_state_gpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/caffe2/python)
    target_link_libraries(
        caffe2_pybind11_state_gpu caffe2_library caffe2_gpu_library)
    install(TARGETS caffe2_pybind11_state_gpu DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python")
  endif()

  if (MSVC AND CMAKE_GENERATOR MATCHES "Visual Studio")
    # If we are building under windows, we will copy the file from
    # build/caffe2/python/{Debug,Release}/caffe2_pybind11_state.pyd
    # to its parent folder so that we can do in-build execution.
    add_custom_target(windows_python_copy_lib ALL)
    add_dependencies(windows_python_copy_lib caffe2_pybind11_state)
    add_custom_command(
        TARGET windows_python_copy_lib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:caffe2_pybind11_state>
        ${CMAKE_BINARY_DIR}/caffe2/python)
    if (USE_CUDA)
      add_dependencies(windows_python_copy_lib caffe2_pybind11_state_gpu)
      add_custom_command(
          TARGET windows_python_copy_lib POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:caffe2_pybind11_state_gpu>
          ${CMAKE_BINARY_DIR}/caffe2/python)
    endif()
  endif()

  # Finally, Copy all python files to build directory
  # Generate and create all needed __init__.py files, if they aren't already
  # present in the current source tree.
  message(STATUS "Automatically generating missing __init__.py files.")
  caffe_autogen_init_py_files()

  # Create a custom target that copies all python files.
  file(GLOB_RECURSE PYTHON_SRCS RELATIVE ${PROJECT_SOURCE_DIR}
       "${PROJECT_SOURCE_DIR}/caffe2/*.py")
  add_custom_target(python_copy_files ALL)
  if(MSVC OR CMAKE_GENERATOR MATCHES "Ninja")
    # ninja fails when the command line is too long so we split
    # the target into several. This would be beneficial for VS also
    # since it build targets in parallel but not custom commands
    foreach(python_src ${PYTHON_SRCS})
      get_filename_component(dir ${python_src} DIRECTORY)
      string(SHA1 name_hash "${python_src}")
      # get_filename_component(name_we ${python_src} NAME_WE)
      add_custom_target(python_copy_files_${name_hash}
          COMMAND ${CMAKE_COMMAND} -E copy
          ${PROJECT_SOURCE_DIR}/${python_src} ${CMAKE_BINARY_DIR}/${dir})
      add_dependencies(python_copy_files python_copy_files_${name_hash})
    endforeach()
  else()
    foreach(python_src ${PYTHON_SRCS})
      get_filename_component(dir ${python_src} DIRECTORY)
      add_custom_command(
          TARGET python_copy_files PRE_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          ${PROJECT_SOURCE_DIR}/${python_src} ${CMAKE_BINARY_DIR}/${dir})
    endforeach()
  endif()

  # Install commands
  # Pick up static python files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe2 DESTINATION ${PYTHON_LIB_REL_PATH}
          FILES_MATCHING PATTERN "*.py")
  # Caffe proto files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe DESTINATION ${PYTHON_LIB_REL_PATH}
          FILES_MATCHING PATTERN "*.py")
  # Caffe2 proto files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe2 DESTINATION ${PYTHON_LIB_REL_PATH}
          FILES_MATCHING PATTERN "*.py")
endif()

# Finally, set the Caffe2_MAIN_LIBS variable in the parent scope.
set(Caffe2_MAIN_LIBS ${Caffe2_MAIN_LIBS} PARENT_SCOPE)
