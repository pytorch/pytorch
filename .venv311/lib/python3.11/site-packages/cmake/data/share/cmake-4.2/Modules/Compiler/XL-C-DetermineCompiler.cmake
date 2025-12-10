
set(_compiler_id_pp_test "defined(__IBMC__) && !defined(__COMPILER_VER__) && __IBMC__ >= 800")

include("${CMAKE_CURRENT_LIST_DIR}/IBMCPP-C-DetermineVersionInternal.cmake")
