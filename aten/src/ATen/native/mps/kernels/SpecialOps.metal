#include <c10/metal/special_math.h>

template <typename T, typename Tout = T>
void kernel
i0(constant T* input,
   device Tout* output,
   uint index [[thread_position_in_grid]]) {
  output[index] = c10::metal::i0(static_cast<Tout>(input[index]));
}

template <typename T, typename Tout = T>
void kernel
i1(constant T* input,
   device Tout* output,
   uint index [[thread_position_in_grid]]) {
  output[index] = c10::metal::i1(static_cast<Tout>(input[index]));
}

#define REGISTER_I0_I1(DTI, DTO)                                        \
  template [[host_name("i0_" #DTI "_" #DTO)]] void kernel i0<DTI, DTO>( \
      constant DTI*, device DTO*, uint);                                \
  template [[host_name("i1_" #DTI "_" #DTO)]] void kernel i1<DTI, DTO>( \
      constant DTI*, device DTO*, uint)

REGISTER_I0_I1(float, float);
REGISTER_I0_I1(bool, float);
REGISTER_I0_I1(uchar, float);
REGISTER_I0_I1(char, float);
REGISTER_I0_I1(short, float);
REGISTER_I0_I1(int, float);
REGISTER_I0_I1(long, float);

REGISTER_I0_I1(half, half);
#if __METAL_VERSION__ >= 310
REGISTER_I0_I1(bfloat, bfloat);
#endif
