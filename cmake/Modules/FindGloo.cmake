# Try to find the Gloo library and headers.
#  Gloo_FOUND        - system has Gloo lib
#  Gloo_INCLUDE_DIRS - the Gloo include directory
#  Gloo_LIBRARIES    - libraries needed to use Gloo

find_path(Gloo_INCLUDE_DIR
	NAMES gloo/common/common.h
	DOC "The directory where Gloo includes reside"
)

find_library(Gloo_NATIVE_LIBRARY
	NAMES gloo
	DOC "The Gloo library (without CUDA)"
)

find_library(Gloo_CUDA_LIBRARY
	NAMES gloo_cuda
	DOC "The Gloo library (with CUDA)"
)

set(Gloo_INCLUDE_DIRS ${Gloo_INCLUDE_DIR})

# use the CUDA library depending on the Gloo_USE_CUDA variable
if (DEFINED Gloo_USE_CUDA)
	if (${Gloo_USE_CUDA})
		set(Gloo_LIBRARY ${Gloo_CUDA_LIBRARY})
	else()
		set(Gloo_LIBRARY ${Gloo_NATIVE_LIBRARY})
	endif()
else()
	# else try to use the CUDA library if found
	if (${Gloo_CUDA_LIBRARY} STREQUAL "Gloo_CUDA_LIBRARY-NOTFOUND")
		set(Gloo_LIBRARY ${Gloo_NATIVE_LIBRARY})
	else()
		set(Gloo_LIBRARY ${Gloo_CUDA_LIBRARY})
	endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gloo
	FOUND_VAR Gloo_FOUND
	REQUIRED_VARS Gloo_INCLUDE_DIR Gloo_LIBRARY
)

mark_as_advanced(Gloo_FOUND)
