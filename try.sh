INC="-I/usr/include/python3.12/"

OPT="-D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_PYTORCH_QNNPACK -DAT_BUILD_ARM_VEC256_WITH_SLEEF -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DHAVE_SVE_CPU_DEFINITION -DHAVE_SVE256_CPU_DEFINITION -DNDEBUG -DNDEBUG -DUSE_MPS"

WARN=" -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-error=dangling-reference -Wno-error=redundant-move  -Wno-stringop-overflow"

/usr/bin/c++ $INC $OPT $WARN -O3 -rdynamic -Wl,--no-as-needed caffe2/torch/lib/libshm/CMakeFiles/torch_shm_manager.dir/manager.cpp.o -o bin/torch_shm_manager  -Wl,-rpath,/Users/kevinpouget/remoting/pytorch/src/build/lib:  lib/libshm.so  -lrt  lib/libc10.so  -Wl,-rpath-link,/Users/kevinpouget/remoting/pytorch/src/build/lib && /usr/local/lib64/python3.12/site-packages/cmake/data/bin/cmake -E __run_co_compile --lwyu="ldd;-u;-r" --source=bin/torch_shm_manager
