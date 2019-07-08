#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()

// needs to be included only once in library.
#include <ideep_pin_singletons.hpp>

#endif // AT_MKLDNN_ENALBED()
