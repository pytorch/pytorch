# Try to find the Gloo library and headers.
#  Gloo_FOUND        - system has Gloo lib
#  Gloo_INCLUDE_DIRS - the Gloo include directory
#  Gloo_NATIVE_LIBRARY - base gloo library, needs to be linked
#  Gloo_CUDA_LIBRARY/Gloo_HIP_LIBRARY - CUDA/HIP support library in Gloo

find_path(Gloo_INCLUDE_DIR
  NAMES gloo/common/common.h
  DOC "The directory where Gloo includes reside"
)

find_library(Gloo_NATIVE_LIBRARY
  NAMES gloo
  DOC "The Gloo library"
)

# Gloo has optional CUDA support
# if Gloo + CUDA is desired, Gloo_CUDA_LIBRARY
# needs to be linked into desired target
find_library(Gloo_CUDA_LIBRARY
  NAMES gloo_cuda
  DOC "Gloo's CUDA support/code"
)

# Gloo has optional HIP support
# if Gloo + HIP is desired, Gloo_HIP_LIBRARY
# needs to be linked to desired target
find_library(Gloo_HIP_LIBRARY
  NAMES gloo_hip
  DOC "Gloo's HIP support/code"
)

set(Gloo_INCLUDE_DIRS ${Gloo_INCLUDE_DIR})


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gloo
  FOUND_VAR Gloo_FOUND
  REQUIRED_VARS Gloo_INCLUDE_DIR Gloo_NATIVE_LIBRARY
)

mark_as_advanced(Gloo_FOUND)
