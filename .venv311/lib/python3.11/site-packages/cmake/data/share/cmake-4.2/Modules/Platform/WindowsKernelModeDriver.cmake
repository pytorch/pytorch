include(Platform/Windows)
macro(__windows_kernel_mode lang)
  if(CMAKE_CROSSCOMPILING)
    set(_KMDF_ERROR_EPILOGUE
        "Please set a valid CMAKE_WINDOWS_KMDF_VERSION in the toolchain file.  "
        "For more information, see\n"
        "  https://learn.microsoft.com/en-us/windows-hardware/drivers/wdf/kmdf-version-history"
        )
    if(NOT DEFINED CMAKE_WINDOWS_KMDF_VERSION)
      message(FATAL_ERROR
        "The Kernel-Mode Driver Framework (KMDF) version has not been set.  "
        ${_KMDF_ERROR_EPILOGUE}
        )
    endif()
    if(NOT CMAKE_WINDOWS_KMDF_VERSION MATCHES "^[0-9]\.[0-9]+$")
      message(FATAL_ERROR
        "The Kernel-Mode Driver Framework (KMDF) version is set to an invalid value.  "
        "The expected format is [0-9].[0-9]+. For example, 1.15 or 1.9.  "
        ${_KMDF_ERROR_EPILOGUE}
        )
    endif()

    set(_KMDF_ENV_VARS
      Platform
      WindowsSdkDir
      VCToolsInstallDir
      )
    if(DEFINED ENV{EnterpriseWDK})
      set(_WINDOWS_SDK_VERSION "$ENV{Version_Number}")
      list(APPEND _KMDF_ENV_VARS Version_Number)
    else()
      set(_WINDOWS_SDK_VERSION "$ENV{WindowsSDKLibVersion}")
      list(APPEND _KMDF_ENV_VARS WindowsSDKLibVersion)
    endif()
    foreach(var IN LISTS _KMDF_ENV_VARS)
      if(NOT DEFINED ENV{${var}})
        message(FATAL_ERROR "Required environment variable '${var}' is not defined.")
      endif()
    endforeach()
    unset(_KMDF_ENV_VARS)

    set(_KMDF_PLATFORM "$ENV{Platform}")

    list(APPEND CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES
        $ENV{WindowsSdkDir}/Include/${_WINDOWS_SDK_VERSION}/km
        $ENV{WindowsSdkDir}/Include/${_WINDOWS_SDK_VERSION}/km/crt
        $ENV{WindowsSdkDir}/Include/${_WINDOWS_SDK_VERSION}/shared
        $ENV{WindowsSdkDir}/Include/wdf/kmdf/${CMAKE_WINDOWS_KMDF_VERSION}
        $ENV{VCToolsInstallDir}/include
        )

    list(APPEND CMAKE_RC_STANDARD_INCLUDE_DIRECTORIES
        ${CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES}
        )
    list(REMOVE_DUPLICATES CMAKE_RC_STANDARD_INCLUDE_DIRECTORIES)

    list(APPEND CMAKE_${lang}_STANDARD_LINK_DIRECTORIES
        $ENV{WindowsSdkDir}/Lib/${_WINDOWS_SDK_VERSION}/km/${_KMDF_PLATFORM}
        )

    unset(_KMDF_ERROR_EPILOGUE)
    unset(_KMDF_PLATFORM)
    unset(_WINDOWS_SDK_VERSION)
  endif()
endmacro()
