# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

macro(cmake_cuda_find_toolkit lang lang_var_)
  # This is very similar to FindCUDAToolkit, but somewhat simplified since we can issue fatal errors
  # if we fail and we don't need to account for searching the libraries.

  # For NVCC we can easily deduce the SDK binary directory from the compiler path.
  if(CMAKE_${lang}_COMPILER_ID STREQUAL "NVIDIA")
    set(_CUDA_NVCC_EXECUTABLE "${CMAKE_${lang}_COMPILER}")
  else()
    # Search using CUDAToolkit_ROOT and then CUDA_PATH for equivalence with FindCUDAToolkit.
    # In FindCUDAToolkit CUDAToolkit_ROOT is searched automatically due to being in a find_package().
    # First we search candidate non-default paths to give them priority.
    find_program(_CUDA_NVCC_EXECUTABLE
      NAMES nvcc nvcc.exe
      PATHS ${CUDAToolkit_ROOT}
      ENV CUDAToolkit_ROOT
      ENV CUDA_PATH
      PATH_SUFFIXES bin
      NO_DEFAULT_PATH
      NO_CACHE
    )

    # If we didn't find NVCC, then try the default paths.
    find_program(_CUDA_NVCC_EXECUTABLE
      NAMES nvcc nvcc.exe
      PATH_SUFFIXES bin
      NO_CACHE
    )

    # If the user specified CUDAToolkit_ROOT but nvcc could not be found, this is an error.
    if(NOT _CUDA_NVCC_EXECUTABLE AND (DEFINED CUDAToolkit_ROOT OR DEFINED ENV{CUDAToolkit_ROOT}))
      set(fail_base "Could not find nvcc executable in path specified by")

      if(DEFINED CUDAToolkit_ROOT)
        message(FATAL_ERROR "${fail_base} CUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
      elseif(DEFINED ENV{CUDAToolkit_ROOT})
        message(FATAL_ERROR "${fail_base} environment variable CUDAToolkit_ROOT=$ENV{CUDAToolkit_ROOT}")
      endif()
    endif()

    # CUDAToolkit_ROOT cmake/env variable not specified, try platform defaults.
    #
    # - Linux: /usr/local/cuda-X.Y
    # - macOS: /Developer/NVIDIA/CUDA-X.Y
    # - Windows: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y
    #
    # We will also search the default symlink location /usr/local/cuda first since
    # if CUDAToolkit_ROOT is not specified, it is assumed that the symlinked
    # directory is the desired location.
    if(NOT _CUDA_NVCC_EXECUTABLE)
      if(UNIX)
        if(NOT APPLE)
          set(platform_base "/usr/local/cuda-")
        else()
          set(platform_base "/Developer/NVIDIA/CUDA-")
        endif()
      else()
        set(platform_base "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v")
      endif()

      # Build out a descending list of possible cuda installations, e.g.
      file(GLOB possible_paths "${platform_base}*")
      # Iterate the glob results and create a descending list.
      set(versions)
      foreach(p ${possible_paths})
        # Extract version number from end of string
        string(REGEX MATCH "[0-9][0-9]?\\.[0-9]$" p_version ${p})
        if(IS_DIRECTORY ${p} AND p_version)
          list(APPEND versions ${p_version})
        endif()
      endforeach()

      # Sort numerically in descending order, so we try the newest versions first.
      list(SORT versions COMPARE NATURAL ORDER DESCENDING)

      # With a descending list of versions, populate possible paths to search.
      set(search_paths)
      foreach(v ${versions})
        list(APPEND search_paths "${platform_base}${v}")
      endforeach()

      # Force the global default /usr/local/cuda to the front on Unix.
      if(UNIX)
        list(INSERT search_paths 0 "/usr/local/cuda")
      endif()

      # Now search for nvcc again using the platform default search paths.
      find_program(_CUDA_NVCC_EXECUTABLE
        NAMES nvcc nvcc.exe
        PATHS ${search_paths}
        PATH_SUFFIXES bin
        NO_CACHE
      )

      # We are done with these variables now, cleanup.
      unset(platform_base)
      unset(possible_paths)
      unset(versions)
      unset(search_paths)

      if(NOT _CUDA_NVCC_EXECUTABLE)
        message(FATAL_ERROR "Failed to find nvcc.\nCompiler ${CMAKE_${lang}_COMPILER_ID} requires the CUDA toolkit. Please set the CUDAToolkit_ROOT variable.")
      endif()
    endif()
  endif()

  # Given that NVCC can be provided by multiple different sources (NVIDIA HPC SDK, CUDA Toolkit, distro)
  # each of which has a different layout, we need to extract the CUDA toolkit root from the compiler
  # itself, allowing us to support numerous different scattered toolkit layouts
  execute_process(COMMAND ${_CUDA_NVCC_EXECUTABLE} "-v" "__cmake_determine_cuda"
    OUTPUT_VARIABLE _CUDA_NVCC_OUT ERROR_VARIABLE _CUDA_NVCC_OUT)
  if(_CUDA_NVCC_OUT MATCHES "\\#\\$ TOP=([^\r\n]*)")
    get_filename_component(${lang_var_}TOOLKIT_ROOT "${CMAKE_MATCH_1}" ABSOLUTE)
  else()
    get_filename_component(${lang_var_}TOOLKIT_ROOT "${_CUDA_NVCC_EXECUTABLE}" DIRECTORY)
    get_filename_component(${lang_var_}TOOLKIT_ROOT "${${lang_var_}TOOLKIT_ROOT}" DIRECTORY)
  endif()

  if(_CUDA_NVCC_OUT MATCHES "\\#\\$ NVVMIR_LIBRARY_DIR=([^\r\n]*)")
    get_filename_component(_CUDA_NVVMIR_LIBRARY_DIR "${CMAKE_MATCH_1}" ABSOLUTE)

    #We require the path to end in `/nvvm/libdevice'
    if(_CUDA_NVVMIR_LIBRARY_DIR MATCHES "nvvm/libdevice$")
      get_filename_component(_CUDA_NVVMIR_LIBRARY_DIR "${_CUDA_NVVMIR_LIBRARY_DIR}/../.." ABSOLUTE)
      set(_CUDA_COMPILER_LIBRARY_ROOT_FROM_NVVMIR_LIBRARY_DIR "${_CUDA_NVVMIR_LIBRARY_DIR}")
    endif()

    unset(_CUDA_NVVMIR_LIBRARY_DIR)
    unset(_cuda_nvvmir_dir_name)
  endif()
  unset(_CUDA_NVCC_OUT)

  # In a non-scattered installation the following are equivalent to ${lang_var_}TOOLKIT_ROOT.
  # We first check for a non-scattered installation to prefer it over a scattered installation.

  # ${lang_var_}LIBRARY_ROOT contains the device library.
  if(DEFINED _CUDA_COMPILER_LIBRARY_ROOT_FROM_NVVMIR_LIBRARY_DIR)
    set(${lang_var_}LIBRARY_ROOT "${_CUDA_COMPILER_LIBRARY_ROOT_FROM_NVVMIR_LIBRARY_DIR}")
  elseif(EXISTS "${${lang_var_}TOOLKIT_ROOT}/nvvm/libdevice")
    set(${lang_var_}LIBRARY_ROOT "${${lang_var_}TOOLKIT_ROOT}")
  elseif(CMAKE_SYSROOT_LINK AND EXISTS "${CMAKE_SYSROOT_LINK}/usr/lib/cuda/nvvm/libdevice")
    set(${lang_var_}LIBRARY_ROOT "${CMAKE_SYSROOT_LINK}/usr/lib/cuda")
  elseif(EXISTS "${CMAKE_SYSROOT}/usr/lib/cuda/nvvm/libdevice")
    set(${lang_var_}LIBRARY_ROOT "${CMAKE_SYSROOT}/usr/lib/cuda")
  else()
    message(FATAL_ERROR "Couldn't find CUDA library root.")
  endif()
  unset(_CUDA_COMPILER_LIBRARY_ROOT_FROM_NVVMIR_LIBRARY_DIR)

  # ${lang_var_}TOOLKIT_LIBRARY_ROOT contains the linking stubs necessary for device linking and other low-level library files.
  if(CMAKE_SYSROOT_LINK AND EXISTS "${CMAKE_SYSROOT_LINK}/usr/lib/nvidia-cuda-toolkit/bin/crt/link.stub")
    set(${lang_var_}TOOLKIT_LIBRARY_ROOT "${CMAKE_SYSROOT_LINK}/usr/lib/nvidia-cuda-toolkit")
  elseif(EXISTS "${CMAKE_SYSROOT}/usr/lib/nvidia-cuda-toolkit/bin/crt/link.stub")
    set(${lang_var_}TOOLKIT_LIBRARY_ROOT "${CMAKE_SYSROOT}/usr/lib/nvidia-cuda-toolkit")
  else()
    set(${lang_var_}TOOLKIT_LIBRARY_ROOT "${${lang_var_}TOOLKIT_ROOT}")
  endif()

  # For regular nvcc we the toolkit version is the same as the compiler version and we can parse it from the vendor test output.
  # For Clang we need to invoke nvcc to get version output.
  if(CMAKE_${lang}_COMPILER_ID STREQUAL "Clang")
    execute_process(COMMAND ${_CUDA_NVCC_EXECUTABLE} "--version" OUTPUT_VARIABLE CMAKE_${lang}_COMPILER_ID_OUTPUT)
  endif()

  if(CMAKE_${lang}_COMPILER_ID_OUTPUT MATCHES [=[V([0-9]+\.[0-9]+\.[0-9]+)]=])
    set(${lang_var_}TOOLKIT_VERSION "${CMAKE_MATCH_1}")
  endif()

  # Don't leak variables unnecessarily to user code.
  unset(_CUDA_NVCC_EXECUTABLE)
endmacro()
