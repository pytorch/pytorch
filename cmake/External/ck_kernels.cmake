#
# create INTERFACE target for CK library
#
if(NOT __ck_kernels_included)
  set(__ck_kernels_included TRUE)

  set(ck_kernels_install_dir "${PROJECT_SOURCE_DIR}/torch/lib")

  set(__ck_kernels_version 0.1)

  # create INTERFACE target
  add_library(__ck_kernels_lib INTERFACE)

  if(DEFINED ENV{CK_KERNELS_INSTALLED_PREFIX})
    # Copy .so from $ENV{CK_KERNELS_INSTALLED_PREFIX} into ${ck_kernels_install_dir}
    install(DIRECTORY
            $ENV{CK_KERNELS_INSTALLED_PREFIX}/
            DESTINATION ${ck_kernels_install_dir}
	    )
    set(ck_kernels_install_path "$ENV{CK_KERNELS_INSTALLED_PREFIX}/libck_kernels.so")
    # specify path to CK library
    target_link_libraries(__ck_kernels_lib INTERFACE ${ck_kernels_install_path})
    message(STATUS "Using Preinstalled CK_kernels from $ENV{CK_KERNELS_INSTALLED_PREFIX}; installed at ${ck_kernels_install_dir}")
  elseif(DEFINED ENV{CK_KERNELS_PACKAGE_BASE_URL})
    # get CK commit hash
    execute_process(
        COMMAND git -C ${CMAKE_SOURCE_DIR}/third_party submodule status composable_kernel
        RESULT_VARIABLE result
        OUTPUT_VARIABLE submodule_status
        ERROR_VARIABLE submodule_status_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    if(result EQUAL 0)
        string(REGEX REPLACE "^[ \t]" "" submodule_status ${submodule_status})
        # extract first 8 characters of the commit hash
        string(SUBSTRING "${submodule_status}" 0 8 ck_commit_hash)
    else()
        message(FATAL_ERROR "Failed to get submodule status for composable_kernel.")
    endif()

    set(ck_kernels_package_full_url "$ENV{CK_KERNELS_PACKAGE_BASE_URL}/torch_ck_gen_lib/ck_${ck_commit_hash}/rocm_${ROCM_VERSION_DEV}/libck_kernels.tar.gz")
    set(ck_kernels_install_path "${ck_kernels_install_dir}/libck_kernels.so")

    ExternalProject_Add(ck_kernels_external
      URL "${ck_kernels_package_full_url}"
      # URL_HASH 
      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/ck_kernels_tarball
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/ck_kernels_tarball"
      "${ck_kernels_install_dir}"
      INSTALL_BYPRODUCTS "${ck_kernels_install_path}"
    )
    add_dependencies(__ck_kernels_lib ck_kernels_external)
    message(STATUS "Using CK_kernels from pre-compiled binary ${ck_kernels_package_full_url}; installed at ${ck_kernels_install_dir}")
    # specify path to CK library
    target_link_libraries(__ck_kernels_lib INTERFACE ${ck_kernels_install_path})
  else()
    set(CK_KERNELS_INSTALL_FROM_SOURCE TRUE)
  endif() # DEFINED ENV{CK_KERNELS_INSTALLED_PREFIX}
  
endif() # __ck_kernels_included
