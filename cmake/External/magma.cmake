if(NOT __MAGMA_INCLUDED)

  set(__MAGMA_INCLUDED TRUE)

  set(__MAGMA_INSTALL_DIR "${PROJECT_SOURCE_DIR}/torch")
  add_library(__rocm_magma INTERFACE)

  # MAGMA package information from GitHub Release Pages
  set(__MAGMA_VER "2.9.0")
  set(__MAGMA_MANYLINUX_LIST
      "manylinux_2_28"  # rocm6.3
      "manylinux_2_28"  # rocm6.4
      "manylinux_2_28"  # rocm7.0
      )
  set(__MAGMA_ROCM_LIST
      "rocm6.3.0"
      "rocm6.4.0"
      "rocm7.0.0"
      )
  set(__MAGMA_CI_COMMIT "")
  set(__MAGMA_EXT ".whl")

  file(MAKE_DIRECTORY ${__MAGMA_INSTALL_DIR}/include/magma)

  # If it is INSTALLED
  if(DEFINED ENV{MAGMA_HOME}) 
    
    # Install from /opt/rocm/magma
    ExternalProject_Add(magma_external
        SOURCE_DIR $ENV{MAGMA_HOME}
        INSTALL_DIR   ${__MAGMA_INSTALL_DIR}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        
        INSTALL_COMMAND   ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include/ <INSTALL_DIR>/include/magma
        COMMAND           ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/lib/libmagma.so <INSTALL_DIR>/lib

        BUILD_BYPRODUCTS <INSTALL_DIR>/lib/libmagma.so
        BUILD_BYPRODUCTS <INSTALL_DIR>/include/magma
        )
    add_dependencies(__rocm_magma magma_external)

    message(STATUS  "Installing Preinstalled MAGMA from $ENV{MAGMA_HOME} to ${__MAGMA_INSTALL_DIR}\n"
                    "MAGMA headers installed to ${__MAGMA_INSTALL_DIR}/include/magma")
                    
  # Installing from WHEEL 
  elseif($ENV{CI})

    if(DEFINED ENV{MAGMA_WHEEL_URL})
      set(__MAGMA_WHEEL_URL $ENV{MAGMA_WHEEL_URL})
    else()
      set(__MAGMA_WHEEL_URL "repo.radeon.com/rocm/magma")
    endif()
    
    set(__MAGMA_SYSTEM_ROCM "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}")
    list(GET __MAGMA_ROCM_LIST 0 __MAGMA_ROCM_DEFAULT_STR)

    # Find ROCm version
    string(SUBSTRING ${__MAGMA_ROCM_DEFAULT_STR} 4 -1 __MAGMA_ROCM)
    foreach(MAGMA_ROCM_BUILD_STR IN LISTS __MAGMA_ROCM_LIST)
      string(SUBSTRING ${MAGMA_ROCM_BUILD_STR} 4 -1 MAGMA_ROCM_BUILD)
      if(MAGMA_ROCM_BUILD VERSION_GREATER __MAGMA_SYSTEM_ROCM)
        break()
      endif()
      set(__MAGMA_ROCM ${MAGMA_ROCM_BUILD})
    endforeach()
    
    list(FIND __MAGMA_ROCM_LIST "rocm${__MAGMA_ROCM}" __MAGMA_ROCM_INDEX)
    list(GET __MAGMA_SHA256_LIST ${__MAGMA_ROCM_INDEX} __MAGMA_SHA256)
    list(GET __MAGMA_MANYLINUX_LIST ${__MAGMA_ROCM_INDEX} __MAGMA_MANYLINUX)
    set(__MAGMA_ARCH "manylinux_2_28_x86_64")

    string(CONCAT __MAGMA_FILE "magma-"
                                  "${__MAGMA_VER}+rocm${__MAGMA_ROCM}"
                                  "-py3-none-${__MAGMA_ARCH}"
                                  "${__MAGMA_EXT}") 

    string(CONCAT __MAGMA_URL "${__MAGMA_WHEEL_URL}/"
                                 "${__MAGMA_FILE}")

    message(STATUS "MAGMA full URL=" ${__MAGMA_URL})



    message(STATUS  "Installing MAGMA from wheel file to ${__MAGMA_INSTALL_DIR}\n"
                    "MAGMA headers in ${__MAGMA_INSTALL_DIR}/include/magma")

    ExternalProject_Add(magma_external
        URL           ${__MAGMA_URL}
        SOURCE_DIR    "${CMAKE_CURRENT_BINARY_DIR}/magma_src"
        INSTALL_DIR   ${__MAGMA_INSTALL_DIR}

        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/magma/lib/libmagma.so <INSTALL_DIR>/lib/
        COMMAND         ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/magma/include <INSTALL_DIR>/include/magma

        BUILD_BYPRODUCTS <INSTALL_DIR>/lib/libmagma.so
        BUILD_BYPRODUCTS <INSTALL_DIR>/include/magma
    )
    add_dependencies(__rocm_magma magma_external)


  else()
    message(STATUS "No MAGMA installation found. Installing MAGMA from source.")

    set(MAGMA_VERSION "2.9.0")
    set(MAGMA_REPOSITORY "https://github.com/ROCm/utk-magma.git")
    set(MAGMA_GIT_TAG "883b14194120a021c802e886c456e86ae2aba164")

    # Find ROCm install 
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH})
      set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${ROCM_PATH})
      list(APPEND CMAKE_MODULE_PATH "${ROCM_PATH}/hip/cmake" "${ROCM_PATH}/lib/cmake/hip") 
    else()
      message(ERROR "No ROCm installation detected. Please install ROCm and set ROCM_PATH before building for ROCm.")
    endif()

    # Install MKL if not installed
    if(DEFINED $ENV{MKLROOT})
    
        set(MKLROOT $ENV{MKLROOT})
        message(STATUS "Attempting to install MAGMA using MKL found in $ENV{MKLROOT}.")
        
    else()
        set(MKLROOT "/opt/intel")
        set(MKL_VERSION "2024.2.0")
        message(STATUS "No MKL installation detected. Attempting to install MKL version ${MKL_VERSION} to ${MKLROOT}.")
        
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        file(MAKE_DIRECTORY ${MKLROOT})
        execute_process(COMMAND ${Python3_EXECUTABLE} -mpip install wheel)
        execute_process(COMMAND ${Python3_EXECUTABLE} -mpip download -d . mkl-static==${MKL_VERSION})
        execute_process(COMMAND ${Python3_EXECUTABLE} -m wheel unpack mkl_static-${MKL_VERSION}-py2.py3-none-manylinux1_x86_64.whl)
        execute_process(COMMAND ${Python3_EXECUTABLE} -m wheel unpack mkl_include-${MKL_VERSION}-py2.py3-none-manylinux1_x86_64.whl)
        execute_process(COMMAND mv mkl_static-${MKL_VERSION}/mkl_static-${MKL_VERSION}.data/data/lib ${MKLROOT})
        execute_process(COMMAND mv mkl_include-${MKL_VERSION}/mkl_include-${MKL_VERSION}.data/data/include ${MKLROOT})
            
    endif()


    set(__MAGMA_EXTERN_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/magma")
    set(__MAGMA_INSTALL_DIR "${PROJECT_SOURCE_DIR}/torch")

    set(target_gpus $ENV{PYTORCH_ROCM_ARCH})
    message(STATUS "MAGMA building for GPU_TARGETS=${target_gpus}")

    cmake_host_system_information(RESULT N_LOGICAL_CORES QUERY NUMBER_OF_LOGICAL_CORES)


    ExternalProject_Add(magma_external
        SOURCE_DIR        ${CMAKE_CURRENT_BINARY_DIR}/magma_src
        PREFIX            ${__MAGMA_EXTERN_PREFIX}
        INSTALL_DIR       ${__MAGMA_INSTALL_DIR}
        GIT_REPOSITORY    ${MAGMA_REPOSITORY}
        GIT_TAG        ${MAGMA_GIT_TAG}

        CONFIGURE_COMMAND  ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/make.inc-examples/make.inc.hip-gcc-mkl <SOURCE_DIR>/make.inc
        COMMAND            ${CMAKE_COMMAND} -E chdir <SOURCE_DIR> make -f make.gen.hipMAGMA -j ${N_LOGICAL_CORES}
        
        BUILD_COMMAND ${CMAKE_COMMAND} -E env MKLROOT=${MKLROOT}
                      ${CMAKE_COMMAND} -E chdir <SOURCE_DIR> make lib/libmagma.so -j ${N_LOGICAL_CORES} MKLROOT=${MKLROOT} GPU_TARGET=${target_gpus}

        INSTALL_COMMAND  ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/lib/libmagma.so <INSTALL_DIR>/lib/
        COMMAND          ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include <INSTALL_DIR>/include/magma
        
        USES_TERMINAL_DOWNLOAD TRUE
        USES_TERMINAL_CONFIGURE TRUE
        USES_TERMINAL_BUILD TRUE
        USES_TERMINAL_INSTALL TRUE
        BUILD_BYPRODUCTS <INSTALL_DIR>/lib/libmagma.so
        BUILD_BYPRODUCTS <INSTALL_DIR>/include/magma
    )
    add_dependencies(__rocm_magma magma_external)
    
    
  endif()
  target_link_libraries(__rocm_magma INTERFACE ${__MAGMA_INSTALL_DIR}/lib/libmagma.so)
  target_include_directories(__rocm_magma INTERFACE ${__MAGMA_INSTALL_DIR}/include/magma)
  set(MAGMA_FOUND TRUE)
endif() # __MAGMA_INCLUDED
