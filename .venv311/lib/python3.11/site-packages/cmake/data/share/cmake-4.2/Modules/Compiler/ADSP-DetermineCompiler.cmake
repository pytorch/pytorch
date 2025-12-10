
set(_compiler_id_pp_test "defined(_ADI_COMPILER)")

set(_compiler_id_version_compute "
#if defined(__VERSIONNUM__)
  /* __VERSIONNUM__ = 0xVVRRPPTT */
#  define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__VERSIONNUM__ >> 24 & 0xFF)
#  define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__VERSIONNUM__ >> 16 & 0xFF)
#  define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__VERSIONNUM__ >> 8 & 0xFF)
#  define @PREFIX@COMPILER_VERSION_TWEAK @MACRO_DEC@(__VERSIONNUM__ & 0xFF)
#endif")
