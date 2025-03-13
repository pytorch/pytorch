#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
using namespace c10::metal;

DEFINE_UNARY_FLOATING_FUNCTOR(bessel_j0_forward);
DEFINE_UNARY_FLOATING_FUNCTOR(bessel_j1_forward);
DEFINE_UNARY_FLOATING_FUNCTOR(bessel_y0_forward);
DEFINE_UNARY_FLOATING_FUNCTOR(bessel_y1_forward);
DEFINE_UNARY_FLOATING_FUNCTOR(i0);
DEFINE_UNARY_FLOATING_FUNCTOR(i1);
DEFINE_UNARY_FLOATING_FUNCTOR(spherical_bessel_j0);
DEFINE_UNARY_FLOATING_FUNCTOR(entr);

#define REGISTER_SPECIAL(DTI, DTO)                  \
  REGISTER_UNARY_OP(bessel_j0_forward, DTI, DTO);   \
  REGISTER_UNARY_OP(bessel_j1_forward, DTI, DTO);   \
  REGISTER_UNARY_OP(bessel_y0_forward, DTI, DTO);   \
  REGISTER_UNARY_OP(bessel_y1_forward, DTI, DTO);   \
  REGISTER_UNARY_OP(i0, DTI, DTO);                  \
  REGISTER_UNARY_OP(i1, DTI, DTO);                  \
  REGISTER_UNARY_OP(spherical_bessel_j0, DTI, DTO); \
  REGISTER_UNARY_OP(entr, DTI, DTO)

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
