set(_compiler_id_pp_test "defined(__RENESAS__)")

set(_compiler_id_version_compute "
/* __RENESAS_VERSION__ = 0xVVRRPP00 */
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_HEX@(__RENESAS_VERSION__ >> 24 & 0xFF)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_HEX@(__RENESAS_VERSION__ >> 16 & 0xFF)
# define @PREFIX@COMPILER_VERSION_PATCH @MACRO_HEX@(__RENESAS_VERSION__ >> 8  & 0xFF)")
