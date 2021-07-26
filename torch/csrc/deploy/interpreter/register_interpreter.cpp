#include <cstdio>

// LIB_START/LIB_END set by -D on gcc cmd line, differently
// depending on cpu or cuda lib

// this lives inside deploy.cpp, which is in libtorch_deploy.so
// we're calling it from libembedded_interpreter.so, hence, across lib
// boundaries
extern "C" __attribute__((__visibility__("default"))) void
register_embedded_interpreter(char* lib_start, char* lib_end);
extern "C" __attribute__((__visibility__("default"))) void
register_embedded_interpreter_cuda(char* lib_start, char* lib_end);

// these should be since we're building this file and linking it
// directly to the embedded_interpreter.a
extern char LIB_START;
extern char LIB_END;

// when libembedded_interpreter.so is opened, we need to register its local
// symbols with torchdeploy
__attribute__((constructor)) static void libconstructor() {
#ifdef EMBEDDED_INTERP_CUDA
  register_embedded_interpreter_cuda(&LIB_START, &LIB_END);
#else
  register_embedded_interpreter(&LIB_START, &LIB_END);
#endif
}
