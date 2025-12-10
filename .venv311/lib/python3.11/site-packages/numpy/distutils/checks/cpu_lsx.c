#ifndef __loongarch_sx
#error "HOST/ARCH doesn't support LSX"
#endif

#include <lsxintrin.h>

int main(void)
{
    __m128i a = __lsx_vadd_d(__lsx_vldi(0), __lsx_vldi(0));
    return __lsx_vpickve2gr_w(a, 0);
}
