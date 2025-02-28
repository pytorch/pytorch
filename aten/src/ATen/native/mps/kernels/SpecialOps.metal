#include <c10/metal/special_math.h>

template <typename T0, typename T1>
void kernel
i0(device T0* output,
   constant T1* input,
   uint index [[thread_position_in_grid]]) {
  output[index] = c10::metal::i0(static_cast<T0>(input[index]));
}

template <typename T0, typename T1>
void kernel
i1(device T0* output,
   constant T1* input,
   uint index [[thread_position_in_grid]]) {
  output[index] = c10::metal::i1(static_cast<T0>(input[index]));
}

template <typename T0, typename T1>
void kernel spherical_bessel_j0(
    device T0* output,
    constant T1* input,
    uint index [[thread_position_in_grid]]) {
  output[index] =
      c10::metal::spherical_bessel_j0(static_cast<T0>(input[index]));
}

template <typename T0, typename T1>
void kernel entr(
    device T0* output,
    constant T1* input,
    uint index [[thread_position_in_grid]]) {
  output[index] = c10::metal::entr(static_cast<T0>(input[index]));
}

#define REGISTER_SPECIAL(DTI, DTO)                                          \
  template [[host_name("i0_" #DTO "_" #DTI)]] void kernel i0<DTO, DTI>(     \
      device DTO*, constant DTI*, uint);                                    \
  template [[host_name("i1_" #DTO "_" #DTI)]] void kernel i1<DTO, DTI>(     \
      device DTO*, constant DTI*, uint);                                    \
  template [[host_name("spherical_bessel_j0_" #DTO "_" #DTI)]] void kernel  \
  spherical_bessel_j0<DTO, DTI>(device DTO*, constant DTI*, uint);          \
  template [[host_name("entr_" #DTO "_" #DTI)]] void kernel entr<DTO, DTI>( \
      device DTO*, constant DTI*, uint);

REGISTER_SPECIAL(float, float);
REGISTER_SPECIAL(bool, float);
REGISTER_SPECIAL(uchar, float);
REGISTER_SPECIAL(char, float);
REGISTER_SPECIAL(short, float);
REGISTER_SPECIAL(int, float);
REGISTER_SPECIAL(long, float);

REGISTER_SPECIAL(half, half);
#if __METAL_VERSION__ >= 310
REGISTER_SPECIAL(bfloat, bfloat);
#endif
