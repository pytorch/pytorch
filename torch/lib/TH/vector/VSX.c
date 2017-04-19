#ifdef __PPC64__
#include <altivec.h>
#include <stddef.h>


//--------------------------------------------------------------------------------------------------
// THDoubleVector_fill_VSX:
//--------------------------------------------------------------------------------------------------
static void THDoubleVector_fill_VSX(double *x, const double c, const ptrdiff_t n)
{
    ptrdiff_t i;

    double val[2] = {c, c};
    vector double fp64vec2 = vec_xl(0, val);

    for (i = 0; i <= n-128; i += 128)
    {
        vec_xst(fp64vec2, 0, x+(i    ));
        vec_xst(fp64vec2, 0, x+(i+2  ));
        vec_xst(fp64vec2, 0, x+(i+4  ));
        vec_xst(fp64vec2, 0, x+(i+6  ));
        vec_xst(fp64vec2, 0, x+(i+8  ));
        vec_xst(fp64vec2, 0, x+(i+10 ));
        vec_xst(fp64vec2, 0, x+(i+12 ));
        vec_xst(fp64vec2, 0, x+(i+14 ));
        vec_xst(fp64vec2, 0, x+(i+16 ));
        vec_xst(fp64vec2, 0, x+(i+18 ));
        vec_xst(fp64vec2, 0, x+(i+20 ));
        vec_xst(fp64vec2, 0, x+(i+22 ));
        vec_xst(fp64vec2, 0, x+(i+24 ));
        vec_xst(fp64vec2, 0, x+(i+26 ));
        vec_xst(fp64vec2, 0, x+(i+28 ));
        vec_xst(fp64vec2, 0, x+(i+30 ));
        vec_xst(fp64vec2, 0, x+(i+32 ));
        vec_xst(fp64vec2, 0, x+(i+34 ));
        vec_xst(fp64vec2, 0, x+(i+36 ));
        vec_xst(fp64vec2, 0, x+(i+38 ));
        vec_xst(fp64vec2, 0, x+(i+40 ));
        vec_xst(fp64vec2, 0, x+(i+42 ));
        vec_xst(fp64vec2, 0, x+(i+44 ));
        vec_xst(fp64vec2, 0, x+(i+46 ));
        vec_xst(fp64vec2, 0, x+(i+48 ));
        vec_xst(fp64vec2, 0, x+(i+50 ));
        vec_xst(fp64vec2, 0, x+(i+52 ));
        vec_xst(fp64vec2, 0, x+(i+54 ));
        vec_xst(fp64vec2, 0, x+(i+56 ));
        vec_xst(fp64vec2, 0, x+(i+58 ));
        vec_xst(fp64vec2, 0, x+(i+60 ));
        vec_xst(fp64vec2, 0, x+(i+62 ));
        vec_xst(fp64vec2, 0, x+(i+64 ));
        vec_xst(fp64vec2, 0, x+(i+66 ));
        vec_xst(fp64vec2, 0, x+(i+68 ));
        vec_xst(fp64vec2, 0, x+(i+70 ));
        vec_xst(fp64vec2, 0, x+(i+72 ));
        vec_xst(fp64vec2, 0, x+(i+74 ));
        vec_xst(fp64vec2, 0, x+(i+76 ));
        vec_xst(fp64vec2, 0, x+(i+78 ));
        vec_xst(fp64vec2, 0, x+(i+80 ));
        vec_xst(fp64vec2, 0, x+(i+82 ));
        vec_xst(fp64vec2, 0, x+(i+84 ));
        vec_xst(fp64vec2, 0, x+(i+86 ));
        vec_xst(fp64vec2, 0, x+(i+88 ));
        vec_xst(fp64vec2, 0, x+(i+90 ));
        vec_xst(fp64vec2, 0, x+(i+92 ));
        vec_xst(fp64vec2, 0, x+(i+94 ));
        vec_xst(fp64vec2, 0, x+(i+96 ));
        vec_xst(fp64vec2, 0, x+(i+98 ));
        vec_xst(fp64vec2, 0, x+(i+100));
        vec_xst(fp64vec2, 0, x+(i+102));
        vec_xst(fp64vec2, 0, x+(i+104));
        vec_xst(fp64vec2, 0, x+(i+106));
        vec_xst(fp64vec2, 0, x+(i+108));
        vec_xst(fp64vec2, 0, x+(i+110));
        vec_xst(fp64vec2, 0, x+(i+112));
        vec_xst(fp64vec2, 0, x+(i+114));
        vec_xst(fp64vec2, 0, x+(i+116));
        vec_xst(fp64vec2, 0, x+(i+118));
        vec_xst(fp64vec2, 0, x+(i+120));
        vec_xst(fp64vec2, 0, x+(i+122));
        vec_xst(fp64vec2, 0, x+(i+124));
        vec_xst(fp64vec2, 0, x+(i+126));
    }
    for (; i <= n-16; i += 16)
    {
        vec_xst(fp64vec2, 0, x+(i    ));
        vec_xst(fp64vec2, 0, x+(i+2  ));
        vec_xst(fp64vec2, 0, x+(i+4  ));
        vec_xst(fp64vec2, 0, x+(i+6  ));
        vec_xst(fp64vec2, 0, x+(i+8  ));
        vec_xst(fp64vec2, 0, x+(i+10 ));
        vec_xst(fp64vec2, 0, x+(i+12 ));
        vec_xst(fp64vec2, 0, x+(i+14 ));
    }
    for (; i <= n-2; i += 2)
        vec_xst(fp64vec2, 0, x+(i    ));
    for (; i < n; i++)
        x[i] = c;
}


//--------------------------------------------------------------------------------------------------
// THDoubleVector_cadds_VSX:
//--------------------------------------------------------------------------------------------------
static void THDoubleVector_cadd_VSX(double *z, const double *x, const double *y, const double c, const ptrdiff_t n)
{
    ptrdiff_t i;

    double val[2] = {c, c};
    vector double c_fp64vec2 = vec_xl(0, val);

    vector double y0_fp64vec2, y1_fp64vec2, y2_fp64vec2, y3_fp64vec2, y4_fp64vec2, y5_fp64vec2, y6_fp64vec2, y7_fp64vec2;
    vector double y8_fp64vec2, y9_fp64vec2, y10_fp64vec2, y11_fp64vec2;
    vector double x0_fp64vec2, x1_fp64vec2, x2_fp64vec2, x3_fp64vec2, x4_fp64vec2, x5_fp64vec2, x6_fp64vec2, x7_fp64vec2;
    vector double x8_fp64vec2, x9_fp64vec2, x10_fp64vec2, x11_fp64vec2;


    for (i = 0; i <= n-24; i += 24)
    {
        y0_fp64vec2  = vec_xl(0, y+(i   ));
        y1_fp64vec2  = vec_xl(0, y+(i+2 ));
        y2_fp64vec2  = vec_xl(0, y+(i+4 ));
        y3_fp64vec2  = vec_xl(0, y+(i+6 ));
        y4_fp64vec2  = vec_xl(0, y+(i+8 ));
        y5_fp64vec2  = vec_xl(0, y+(i+10));
        y6_fp64vec2  = vec_xl(0, y+(i+12));
        y7_fp64vec2  = vec_xl(0, y+(i+14));
        y8_fp64vec2  = vec_xl(0, y+(i+16));
        y9_fp64vec2  = vec_xl(0, y+(i+18));
        y10_fp64vec2 = vec_xl(0, y+(i+20));
        y11_fp64vec2 = vec_xl(0, y+(i+22));

        x0_fp64vec2  = vec_xl(0, x+(i   ));
        x1_fp64vec2  = vec_xl(0, x+(i+2 ));
        x2_fp64vec2  = vec_xl(0, x+(i+4 ));
        x3_fp64vec2  = vec_xl(0, x+(i+6 ));
        x4_fp64vec2  = vec_xl(0, x+(i+8 ));
        x5_fp64vec2  = vec_xl(0, x+(i+10));
        x6_fp64vec2  = vec_xl(0, x+(i+12));
        x7_fp64vec2  = vec_xl(0, x+(i+14));
        x8_fp64vec2  = vec_xl(0, x+(i+16));
        x9_fp64vec2  = vec_xl(0, x+(i+18));
        x10_fp64vec2 = vec_xl(0, x+(i+20));
        x11_fp64vec2 = vec_xl(0, x+(i+22));

        y0_fp64vec2  = vec_madd(y0_fp64vec2, c_fp64vec2,  x0_fp64vec2);
        y1_fp64vec2  = vec_madd(y1_fp64vec2, c_fp64vec2, x1_fp64vec2);
        y2_fp64vec2  = vec_madd(y2_fp64vec2, c_fp64vec2, x2_fp64vec2);
        y3_fp64vec2  = vec_madd(y3_fp64vec2, c_fp64vec2, x3_fp64vec2);
        y4_fp64vec2  = vec_madd(y4_fp64vec2, c_fp64vec2, x4_fp64vec2);
        y5_fp64vec2  = vec_madd(y5_fp64vec2, c_fp64vec2, x5_fp64vec2);
        y6_fp64vec2  = vec_madd(y6_fp64vec2, c_fp64vec2, x6_fp64vec2);
        y7_fp64vec2  = vec_madd(y7_fp64vec2, c_fp64vec2, x7_fp64vec2);
        y8_fp64vec2  = vec_madd(y8_fp64vec2, c_fp64vec2, x8_fp64vec2);
        y9_fp64vec2  = vec_madd(y9_fp64vec2, c_fp64vec2, x9_fp64vec2);
        y10_fp64vec2 = vec_madd(y10_fp64vec2, c_fp64vec2,x10_fp64vec2);
        y11_fp64vec2 = vec_madd(y11_fp64vec2, c_fp64vec2,x11_fp64vec2);

        vec_xst(y0_fp64vec2,  0, z+(i   ));
        vec_xst(y1_fp64vec2,  0, z+(i+2 ));
        vec_xst(y2_fp64vec2,  0, z+(i+4 ));
        vec_xst(y3_fp64vec2,  0, z+(i+6 ));
        vec_xst(y4_fp64vec2,  0, z+(i+8 ));
        vec_xst(y5_fp64vec2,  0, z+(i+10));
        vec_xst(y6_fp64vec2,  0, z+(i+12));
        vec_xst(y7_fp64vec2,  0, z+(i+14));
        vec_xst(y8_fp64vec2,  0, z+(i+16));
        vec_xst(y9_fp64vec2,  0, z+(i+18));
        vec_xst(y10_fp64vec2, 0, z+(i+20));
        vec_xst(y11_fp64vec2, 0, z+(i+22));
    }
    for (; i <= n-8; i += 8)
    {
        y0_fp64vec2  = vec_xl(0, y+(i   ));
        y1_fp64vec2  = vec_xl(0, y+(i+2 ));
        y2_fp64vec2  = vec_xl(0, y+(i+4 ));
        y3_fp64vec2  = vec_xl(0, y+(i+6 ));

        x0_fp64vec2  = vec_xl(0, x+(i   ));
        x1_fp64vec2  = vec_xl(0, x+(i+2 ));
        x2_fp64vec2  = vec_xl(0, x+(i+4 ));
        x3_fp64vec2  = vec_xl(0, x+(i+6 ));

        y0_fp64vec2  = vec_madd(y0_fp64vec2, c_fp64vec2, x0_fp64vec2);
        y1_fp64vec2  = vec_madd(y1_fp64vec2, c_fp64vec2, x1_fp64vec2);
        y2_fp64vec2  = vec_madd(y2_fp64vec2, c_fp64vec2, x2_fp64vec2);
        y3_fp64vec2  = vec_madd(y3_fp64vec2, c_fp64vec2, x3_fp64vec2);

        vec_xst(y0_fp64vec2,  0, z+(i   ));
        vec_xst(y1_fp64vec2,  0, z+(i+2 ));
        vec_xst(y2_fp64vec2,  0, z+(i+4 ));
        vec_xst(y3_fp64vec2,  0, z+(i+6 ));
    }
    for (; i <= n-2; i += 2)
    {
        y0_fp64vec2  = vec_xl(0, y+(i   ));
        x0_fp64vec2  = vec_xl(0, x+(i   ));
        y0_fp64vec2  = vec_madd(y0_fp64vec2, c_fp64vec2, x0_fp64vec2);
        vec_xst(y0_fp64vec2,  0, z+(i   ));
    }
    for (; i < n; i++)
        z[i] = x[i] + c* y[i];
}


//--------------------------------------------------------------------------------------------------
// THDoubleVector_adds_VSX:
//--------------------------------------------------------------------------------------------------
static void THDoubleVector_adds_VSX(double *y, const double *x, const double c, const ptrdiff_t n)
{
    ptrdiff_t i;

    double val[2] = {c, c};
    vector double c_fp64vec2 = vec_xl(0, val);

    vector double y0_fp64vec2, y1_fp64vec2, y2_fp64vec2, y3_fp64vec2, y4_fp64vec2, y5_fp64vec2, y6_fp64vec2, y7_fp64vec2;
    vector double y8_fp64vec2, y9_fp64vec2, y10_fp64vec2, y11_fp64vec2;
    vector double x0_fp64vec2, x1_fp64vec2, x2_fp64vec2, x3_fp64vec2, x4_fp64vec2, x5_fp64vec2, x6_fp64vec2, x7_fp64vec2;
    vector double x8_fp64vec2, x9_fp64vec2, x10_fp64vec2, x11_fp64vec2;


    for (i = 0; i <= n-24; i += 24)
    {
        x0_fp64vec2  = vec_xl(0, x+(i   ));
        x1_fp64vec2  = vec_xl(0, x+(i+2 ));
        x2_fp64vec2  = vec_xl(0, x+(i+4 ));
        x3_fp64vec2  = vec_xl(0, x+(i+6 ));
        x4_fp64vec2  = vec_xl(0, x+(i+8 ));
        x5_fp64vec2  = vec_xl(0, x+(i+10));
        x6_fp64vec2  = vec_xl(0, x+(i+12));
        x7_fp64vec2  = vec_xl(0, x+(i+14));
        x8_fp64vec2  = vec_xl(0, x+(i+16));
        x9_fp64vec2  = vec_xl(0, x+(i+18));
        x10_fp64vec2 = vec_xl(0, x+(i+20));
        x11_fp64vec2 = vec_xl(0, x+(i+22));

        y0_fp64vec2  = vec_add(x0_fp64vec2,  c_fp64vec2);
        y1_fp64vec2  = vec_add(x1_fp64vec2,  c_fp64vec2);
        y2_fp64vec2  = vec_add(x2_fp64vec2,  c_fp64vec2);
        y3_fp64vec2  = vec_add(x3_fp64vec2,  c_fp64vec2);
        y4_fp64vec2  = vec_add(x4_fp64vec2,  c_fp64vec2);
        y5_fp64vec2  = vec_add(x5_fp64vec2,  c_fp64vec2);
        y6_fp64vec2  = vec_add(x6_fp64vec2,  c_fp64vec2);
        y7_fp64vec2  = vec_add(x7_fp64vec2,  c_fp64vec2);
        y8_fp64vec2  = vec_add(x8_fp64vec2,  c_fp64vec2);
        y9_fp64vec2  = vec_add(x9_fp64vec2,  c_fp64vec2);
        y10_fp64vec2 = vec_add(x10_fp64vec2, c_fp64vec2);
        y11_fp64vec2 = vec_add(x11_fp64vec2, c_fp64vec2);
        

        vec_xst(y0_fp64vec2,  0, y+(i   ));
        vec_xst(y1_fp64vec2,  0, y+(i+2 ));
        vec_xst(y2_fp64vec2,  0, y+(i+4 ));
        vec_xst(y3_fp64vec2,  0, y+(i+6 ));
        vec_xst(y4_fp64vec2,  0, y+(i+8 ));
        vec_xst(y5_fp64vec2,  0, y+(i+10));
        vec_xst(y6_fp64vec2,  0, y+(i+12));
        vec_xst(y7_fp64vec2,  0, y+(i+14));
        vec_xst(y8_fp64vec2,  0, y+(i+16));
        vec_xst(y9_fp64vec2,  0, y+(i+18));
        vec_xst(y10_fp64vec2, 0, y+(i+20));
        vec_xst(y11_fp64vec2, 0, y+(i+22));
    }
    for (; i <= n-8; i += 8)
    {
        x0_fp64vec2  = vec_xl(0, x+(i   ));
        x1_fp64vec2  = vec_xl(0, x+(i+2 ));
        x2_fp64vec2  = vec_xl(0, x+(i+4 ));
        x3_fp64vec2  = vec_xl(0, x+(i+6 ));

        y0_fp64vec2  = vec_add(x0_fp64vec2,  c_fp64vec2);
        y1_fp64vec2  = vec_add(x1_fp64vec2,  c_fp64vec2);
        y2_fp64vec2  = vec_add(x2_fp64vec2,  c_fp64vec2);
        y3_fp64vec2  = vec_add(x3_fp64vec2,  c_fp64vec2);

        vec_xst(y0_fp64vec2,  0, y+(i   ));
        vec_xst(y1_fp64vec2,  0, y+(i+2 ));
        vec_xst(y2_fp64vec2,  0, y+(i+4 ));
        vec_xst(y3_fp64vec2,  0, y+(i+6 ));
    }
    for (; i <= n-2; i += 2)
    {
        x0_fp64vec2  = vec_xl(0, x+(i   ));
        y0_fp64vec2  = vec_add(x0_fp64vec2,  c_fp64vec2);
        vec_xst(y0_fp64vec2,  0, y+(i   ));
    }
    for (; i < n; i++)
        y[i] = x[i] +c;
}


//--------------------------------------------------------------------------------------------------
// THDoubleVector_cmul_VSX:
//--------------------------------------------------------------------------------------------------
static void THDoubleVector_cmul_VSX(double *z, const double *x, const double *y, const ptrdiff_t n)
{
    ptrdiff_t i;

    vector double y0_fp64vec2, y1_fp64vec2, y2_fp64vec2, y3_fp64vec2, y4_fp64vec2, y5_fp64vec2, y6_fp64vec2, y7_fp64vec2;
    vector double y8_fp64vec2, y9_fp64vec2, y10_fp64vec2, y11_fp64vec2;
    vector double x0_fp64vec2, x1_fp64vec2, x2_fp64vec2, x3_fp64vec2, x4_fp64vec2, x5_fp64vec2, x6_fp64vec2, x7_fp64vec2;
    vector double x8_fp64vec2, x9_fp64vec2, x10_fp64vec2, x11_fp64vec2;


    for (i = 0; i <= n-24; i += 24)
    {
        y0_fp64vec2  = vec_xl(0, y+(i   ));
        y1_fp64vec2  = vec_xl(0, y+(i+2 ));
        y2_fp64vec2  = vec_xl(0, y+(i+4 ));
        y3_fp64vec2  = vec_xl(0, y+(i+6 ));
        y4_fp64vec2  = vec_xl(0, y+(i+8 ));
        y5_fp64vec2  = vec_xl(0, y+(i+10));
        y6_fp64vec2  = vec_xl(0, y+(i+12));
        y7_fp64vec2  = vec_xl(0, y+(i+14));
        y8_fp64vec2  = vec_xl(0, y+(i+16));
        y9_fp64vec2  = vec_xl(0, y+(i+18));
        y10_fp64vec2 = vec_xl(0, y+(i+20));
        y11_fp64vec2 = vec_xl(0, y+(i+22));

        x0_fp64vec2  = vec_xl(0, x+(i   ));
        x1_fp64vec2  = vec_xl(0, x+(i+2 ));
        x2_fp64vec2  = vec_xl(0, x+(i+4 ));
        x3_fp64vec2  = vec_xl(0, x+(i+6 ));
        x4_fp64vec2  = vec_xl(0, x+(i+8 ));
        x5_fp64vec2  = vec_xl(0, x+(i+10));
        x6_fp64vec2  = vec_xl(0, x+(i+12));
        x7_fp64vec2  = vec_xl(0, x+(i+14));
        x8_fp64vec2  = vec_xl(0, x+(i+16));
        x9_fp64vec2  = vec_xl(0, x+(i+18));
        x10_fp64vec2 = vec_xl(0, x+(i+20));
        x11_fp64vec2 = vec_xl(0, x+(i+22));

        y0_fp64vec2  = vec_mul(y0_fp64vec2,  x0_fp64vec2);
        y1_fp64vec2  = vec_mul(y1_fp64vec2,  x1_fp64vec2);
        y2_fp64vec2  = vec_mul(y2_fp64vec2,  x2_fp64vec2);
        y3_fp64vec2  = vec_mul(y3_fp64vec2,  x3_fp64vec2);
        y4_fp64vec2  = vec_mul(y4_fp64vec2,  x4_fp64vec2);
        y5_fp64vec2  = vec_mul(y5_fp64vec2,  x5_fp64vec2);
        y6_fp64vec2  = vec_mul(y6_fp64vec2,  x6_fp64vec2);
        y7_fp64vec2  = vec_mul(y7_fp64vec2,  x7_fp64vec2);
        y8_fp64vec2  = vec_mul(y8_fp64vec2,  x8_fp64vec2);
        y9_fp64vec2  = vec_mul(y9_fp64vec2,  x9_fp64vec2);
        y10_fp64vec2 = vec_mul(y10_fp64vec2, x10_fp64vec2);
        y11_fp64vec2 = vec_mul(y11_fp64vec2, x11_fp64vec2);

        vec_xst(y0_fp64vec2,  0, z+(i   ));
        vec_xst(y1_fp64vec2,  0, z+(i+2 ));
        vec_xst(y2_fp64vec2,  0, z+(i+4 ));
        vec_xst(y3_fp64vec2,  0, z+(i+6 ));
        vec_xst(y4_fp64vec2,  0, z+(i+8 ));
        vec_xst(y5_fp64vec2,  0, z+(i+10));
        vec_xst(y6_fp64vec2,  0, z+(i+12));
        vec_xst(y7_fp64vec2,  0, z+(i+14));
        vec_xst(y8_fp64vec2,  0, z+(i+16));
        vec_xst(y9_fp64vec2,  0, z+(i+18));
        vec_xst(y10_fp64vec2, 0, z+(i+20));
        vec_xst(y11_fp64vec2, 0, z+(i+22));
    }
    for (; i <= n-8; i += 8)
    {
        y0_fp64vec2  = vec_xl(0, y+(i   ));
        y1_fp64vec2  = vec_xl(0, y+(i+2 ));
        y2_fp64vec2  = vec_xl(0, y+(i+4 ));
        y3_fp64vec2  = vec_xl(0, y+(i+6 ));

        x0_fp64vec2  = vec_xl(0, x+(i   ));
        x1_fp64vec2  = vec_xl(0, x+(i+2 ));
        x2_fp64vec2  = vec_xl(0, x+(i+4 ));
        x3_fp64vec2  = vec_xl(0, x+(i+6 ));

        y0_fp64vec2  = vec_mul(y0_fp64vec2,  x0_fp64vec2);
        y1_fp64vec2  = vec_mul(y1_fp64vec2,  x1_fp64vec2);
        y2_fp64vec2  = vec_mul(y2_fp64vec2,  x2_fp64vec2);
        y3_fp64vec2  = vec_mul(y3_fp64vec2,  x3_fp64vec2);

        vec_xst(y0_fp64vec2,  0, z+(i   ));
        vec_xst(y1_fp64vec2,  0, z+(i+2 ));
        vec_xst(y2_fp64vec2,  0, z+(i+4 ));
        vec_xst(y3_fp64vec2,  0, z+(i+6 ));
    }
    for (; i <= n-2; i += 2)
    {
        y0_fp64vec2  = vec_xl(0, y+(i   ));
        x0_fp64vec2  = vec_xl(0, x+(i   ));
        y0_fp64vec2  = vec_mul(y0_fp64vec2,  x0_fp64vec2);
        vec_xst(y0_fp64vec2,  0, z+(i   ));
    }
    for (; i < n; i++)
        z[i] = x[i] * y[i];
}


//--------------------------------------------------------------------------------------------------
// THDoubleVector_muls_VSX:
//--------------------------------------------------------------------------------------------------
static void THDoubleVector_muls_VSX(double *y, const double *x, const double c, const ptrdiff_t n)
{
    ptrdiff_t i;

    double val[2] = {c, c};
    vector double c_fp64vec2 = vec_xl(0, val);

    vector double y0_fp64vec2, y1_fp64vec2, y2_fp64vec2, y3_fp64vec2, y4_fp64vec2, y5_fp64vec2, y6_fp64vec2, y7_fp64vec2;
    vector double y8_fp64vec2, y9_fp64vec2, y10_fp64vec2, y11_fp64vec2;
    vector double x0_fp64vec2, x1_fp64vec2, x2_fp64vec2, x3_fp64vec2, x4_fp64vec2, x5_fp64vec2, x6_fp64vec2, x7_fp64vec2;
    vector double x8_fp64vec2, x9_fp64vec2, x10_fp64vec2, x11_fp64vec2;


    for (i = 0; i <= n-24; i += 24)
    {
        x0_fp64vec2  = vec_xl(0, x+(i   ));
        x1_fp64vec2  = vec_xl(0, x+(i+2 ));
        x2_fp64vec2  = vec_xl(0, x+(i+4 ));
        x3_fp64vec2  = vec_xl(0, x+(i+6 ));
        x4_fp64vec2  = vec_xl(0, x+(i+8 ));
        x5_fp64vec2  = vec_xl(0, x+(i+10));
        x6_fp64vec2  = vec_xl(0, x+(i+12));
        x7_fp64vec2  = vec_xl(0, x+(i+14));
        x8_fp64vec2  = vec_xl(0, x+(i+16));
        x9_fp64vec2  = vec_xl(0, x+(i+18));
        x10_fp64vec2 = vec_xl(0, x+(i+20));
        x11_fp64vec2 = vec_xl(0, x+(i+22));

        y0_fp64vec2  = vec_mul(x0_fp64vec2,  c_fp64vec2);
        y1_fp64vec2  = vec_mul(x1_fp64vec2,  c_fp64vec2);
        y2_fp64vec2  = vec_mul(x2_fp64vec2,  c_fp64vec2);
        y3_fp64vec2  = vec_mul(x3_fp64vec2,  c_fp64vec2);
        y4_fp64vec2  = vec_mul(x4_fp64vec2,  c_fp64vec2);
        y5_fp64vec2  = vec_mul(x5_fp64vec2,  c_fp64vec2);
        y6_fp64vec2  = vec_mul(x6_fp64vec2,  c_fp64vec2);
        y7_fp64vec2  = vec_mul(x7_fp64vec2,  c_fp64vec2);
        y8_fp64vec2  = vec_mul(x8_fp64vec2,  c_fp64vec2);
        y9_fp64vec2  = vec_mul(x9_fp64vec2,  c_fp64vec2);
        y10_fp64vec2 = vec_mul(x10_fp64vec2, c_fp64vec2);
        y11_fp64vec2 = vec_mul(x11_fp64vec2, c_fp64vec2);
        

        vec_xst(y0_fp64vec2,  0, y+(i   ));
        vec_xst(y1_fp64vec2,  0, y+(i+2 ));
        vec_xst(y2_fp64vec2,  0, y+(i+4 ));
        vec_xst(y3_fp64vec2,  0, y+(i+6 ));
        vec_xst(y4_fp64vec2,  0, y+(i+8 ));
        vec_xst(y5_fp64vec2,  0, y+(i+10));
        vec_xst(y6_fp64vec2,  0, y+(i+12));
        vec_xst(y7_fp64vec2,  0, y+(i+14));
        vec_xst(y8_fp64vec2,  0, y+(i+16));
        vec_xst(y9_fp64vec2,  0, y+(i+18));
        vec_xst(y10_fp64vec2, 0, y+(i+20));
        vec_xst(y11_fp64vec2, 0, y+(i+22));
    }
    for (; i <= n-8; i += 8)
    {
        x0_fp64vec2  = vec_xl(0, x+(i   ));
        x1_fp64vec2  = vec_xl(0, x+(i+2 ));
        x2_fp64vec2  = vec_xl(0, x+(i+4 ));
        x3_fp64vec2  = vec_xl(0, x+(i+6 ));

        y0_fp64vec2  = vec_mul(x0_fp64vec2,  c_fp64vec2);
        y1_fp64vec2  = vec_mul(x1_fp64vec2,  c_fp64vec2);
        y2_fp64vec2  = vec_mul(x2_fp64vec2,  c_fp64vec2);
        y3_fp64vec2  = vec_mul(x3_fp64vec2,  c_fp64vec2);

        vec_xst(y0_fp64vec2,  0, y+(i   ));
        vec_xst(y1_fp64vec2,  0, y+(i+2 ));
        vec_xst(y2_fp64vec2,  0, y+(i+4 ));
        vec_xst(y3_fp64vec2,  0, y+(i+6 ));
    }
    for (; i <= n-2; i += 2)
    {
        x0_fp64vec2  = vec_xl(0, x+(i   ));
        y0_fp64vec2  = vec_mul(x0_fp64vec2,  c_fp64vec2);
        vec_xst(y0_fp64vec2,  0, y+(i   ));
    }
    for (; i < n; i++)
        y[i] = c * x[i];
}


//--------------------------------------------------------------------------------------------------
// THDoubleVector_cdiv_VSX:
//--------------------------------------------------------------------------------------------------
static void THDoubleVector_cdiv_VSX(double *z, const double *x, const double *y, const ptrdiff_t n)
{
    ptrdiff_t i;

    vector double y0_fp64vec2, y1_fp64vec2, y2_fp64vec2, y3_fp64vec2, y4_fp64vec2, y5_fp64vec2, y6_fp64vec2, y7_fp64vec2;
    vector double y8_fp64vec2, y9_fp64vec2, y10_fp64vec2, y11_fp64vec2;
    vector double x0_fp64vec2, x1_fp64vec2, x2_fp64vec2, x3_fp64vec2, x4_fp64vec2, x5_fp64vec2, x6_fp64vec2, x7_fp64vec2;
    vector double x8_fp64vec2, x9_fp64vec2, x10_fp64vec2, x11_fp64vec2;


    for (i = 0; i <= n-24; i += 24)
    {
        y0_fp64vec2  = vec_xl(0, y+(i   ));
        y1_fp64vec2  = vec_xl(0, y+(i+2 ));
        y2_fp64vec2  = vec_xl(0, y+(i+4 ));
        y3_fp64vec2  = vec_xl(0, y+(i+6 ));
        y4_fp64vec2  = vec_xl(0, y+(i+8 ));
        y5_fp64vec2  = vec_xl(0, y+(i+10));
        y6_fp64vec2  = vec_xl(0, y+(i+12));
        y7_fp64vec2  = vec_xl(0, y+(i+14));
        y8_fp64vec2  = vec_xl(0, y+(i+16));
        y9_fp64vec2  = vec_xl(0, y+(i+18));
        y10_fp64vec2 = vec_xl(0, y+(i+20));
        y11_fp64vec2 = vec_xl(0, y+(i+22));

        x0_fp64vec2  = vec_xl(0, x+(i   ));
        x1_fp64vec2  = vec_xl(0, x+(i+2 ));
        x2_fp64vec2  = vec_xl(0, x+(i+4 ));
        x3_fp64vec2  = vec_xl(0, x+(i+6 ));
        x4_fp64vec2  = vec_xl(0, x+(i+8 ));
        x5_fp64vec2  = vec_xl(0, x+(i+10));
        x6_fp64vec2  = vec_xl(0, x+(i+12));
        x7_fp64vec2  = vec_xl(0, x+(i+14));
        x8_fp64vec2  = vec_xl(0, x+(i+16));
        x9_fp64vec2  = vec_xl(0, x+(i+18));
        x10_fp64vec2 = vec_xl(0, x+(i+20));
        x11_fp64vec2 = vec_xl(0, x+(i+22));

        y0_fp64vec2  = vec_div(x0_fp64vec2,  y0_fp64vec2);
        y1_fp64vec2  = vec_div(x1_fp64vec2,  y1_fp64vec2);
        y2_fp64vec2  = vec_div(x2_fp64vec2,  y2_fp64vec2);
        y3_fp64vec2  = vec_div(x3_fp64vec2,  y3_fp64vec2);
        y4_fp64vec2  = vec_div(x4_fp64vec2,  y4_fp64vec2);
        y5_fp64vec2  = vec_div(x5_fp64vec2,  y5_fp64vec2);
        y6_fp64vec2  = vec_div(x6_fp64vec2,  y6_fp64vec2);
        y7_fp64vec2  = vec_div(x7_fp64vec2,  y7_fp64vec2);
        y8_fp64vec2  = vec_div(x8_fp64vec2,  y8_fp64vec2);
        y9_fp64vec2  = vec_div(x9_fp64vec2,  y9_fp64vec2);
        y10_fp64vec2 = vec_div(x10_fp64vec2, y10_fp64vec2);
        y11_fp64vec2 = vec_div(x11_fp64vec2, y11_fp64vec2);

        vec_xst(y0_fp64vec2,  0, z+(i   ));
        vec_xst(y1_fp64vec2,  0, z+(i+2 ));
        vec_xst(y2_fp64vec2,  0, z+(i+4 ));
        vec_xst(y3_fp64vec2,  0, z+(i+6 ));
        vec_xst(y4_fp64vec2,  0, z+(i+8 ));
        vec_xst(y5_fp64vec2,  0, z+(i+10));
        vec_xst(y6_fp64vec2,  0, z+(i+12));
        vec_xst(y7_fp64vec2,  0, z+(i+14));
        vec_xst(y8_fp64vec2,  0, z+(i+16));
        vec_xst(y9_fp64vec2,  0, z+(i+18));
        vec_xst(y10_fp64vec2, 0, z+(i+20));
        vec_xst(y11_fp64vec2, 0, z+(i+22));
    }
    for (; i <= n-8; i += 8)
    {
        y0_fp64vec2  = vec_xl(0, y+(i   ));
        y1_fp64vec2  = vec_xl(0, y+(i+2 ));
        y2_fp64vec2  = vec_xl(0, y+(i+4 ));
        y3_fp64vec2  = vec_xl(0, y+(i+6 ));

        x0_fp64vec2  = vec_xl(0, x+(i   ));
        x1_fp64vec2  = vec_xl(0, x+(i+2 ));
        x2_fp64vec2  = vec_xl(0, x+(i+4 ));
        x3_fp64vec2  = vec_xl(0, x+(i+6 ));

        y0_fp64vec2  = vec_div(x0_fp64vec2,  y0_fp64vec2);
        y1_fp64vec2  = vec_div(x1_fp64vec2,  y1_fp64vec2);
        y2_fp64vec2  = vec_div(x2_fp64vec2,  y2_fp64vec2);
        y3_fp64vec2  = vec_div(x3_fp64vec2,  y3_fp64vec2);

        vec_xst(y0_fp64vec2,  0, z+(i   ));
        vec_xst(y1_fp64vec2,  0, z+(i+2 ));
        vec_xst(y2_fp64vec2,  0, z+(i+4 ));
        vec_xst(y3_fp64vec2,  0, z+(i+6 ));
    }
    for (; i <= n-2; i += 2)
    {
        y0_fp64vec2  = vec_xl(0, y+(i   ));
        x0_fp64vec2  = vec_xl(0, x+(i   ));
        y0_fp64vec2  = vec_div(x0_fp64vec2,  y0_fp64vec2);
        vec_xst(y0_fp64vec2,  0, z+(i   ));
    }
    for (; i < n; i++)
        z[i] = x[i] / y[i];
}


//--------------------------------------------------------------------------------------------------
// THDoubleVector_divs_VSX:
//--------------------------------------------------------------------------------------------------
static void THDoubleVector_divs_VSX(double *y, const double *x, const double c, const ptrdiff_t n)
{
    ptrdiff_t i;

    double val[2] = {c, c};
    vector double c_fp64vec2 = vec_xl(0, val);

    vector double y0_fp64vec2, y1_fp64vec2, y2_fp64vec2, y3_fp64vec2, y4_fp64vec2, y5_fp64vec2, y6_fp64vec2, y7_fp64vec2;
    vector double y8_fp64vec2, y9_fp64vec2, y10_fp64vec2, y11_fp64vec2;
    vector double x0_fp64vec2, x1_fp64vec2, x2_fp64vec2, x3_fp64vec2, x4_fp64vec2, x5_fp64vec2, x6_fp64vec2, x7_fp64vec2;
    vector double x8_fp64vec2, x9_fp64vec2, x10_fp64vec2, x11_fp64vec2;


    for (i = 0; i <= n-24; i += 24)
    {
        x0_fp64vec2  = vec_xl(0, x+(i   ));
        x1_fp64vec2  = vec_xl(0, x+(i+2 ));
        x2_fp64vec2  = vec_xl(0, x+(i+4 ));
        x3_fp64vec2  = vec_xl(0, x+(i+6 ));
        x4_fp64vec2  = vec_xl(0, x+(i+8 ));
        x5_fp64vec2  = vec_xl(0, x+(i+10));
        x6_fp64vec2  = vec_xl(0, x+(i+12));
        x7_fp64vec2  = vec_xl(0, x+(i+14));
        x8_fp64vec2  = vec_xl(0, x+(i+16));
        x9_fp64vec2  = vec_xl(0, x+(i+18));
        x10_fp64vec2 = vec_xl(0, x+(i+20));
        x11_fp64vec2 = vec_xl(0, x+(i+22));

        y0_fp64vec2  = vec_div(x0_fp64vec2,  c_fp64vec2);
        y1_fp64vec2  = vec_div(x1_fp64vec2,  c_fp64vec2);
        y2_fp64vec2  = vec_div(x2_fp64vec2,  c_fp64vec2);
        y3_fp64vec2  = vec_div(x3_fp64vec2,  c_fp64vec2);
        y4_fp64vec2  = vec_div(x4_fp64vec2,  c_fp64vec2);
        y5_fp64vec2  = vec_div(x5_fp64vec2,  c_fp64vec2);
        y6_fp64vec2  = vec_div(x6_fp64vec2,  c_fp64vec2);
        y7_fp64vec2  = vec_div(x7_fp64vec2,  c_fp64vec2);
        y8_fp64vec2  = vec_div(x8_fp64vec2,  c_fp64vec2);
        y9_fp64vec2  = vec_div(x9_fp64vec2,  c_fp64vec2);
        y10_fp64vec2 = vec_div(x10_fp64vec2, c_fp64vec2);
        y11_fp64vec2 = vec_div(x11_fp64vec2, c_fp64vec2);
        

        vec_xst(y0_fp64vec2,  0, y+(i   ));
        vec_xst(y1_fp64vec2,  0, y+(i+2 ));
        vec_xst(y2_fp64vec2,  0, y+(i+4 ));
        vec_xst(y3_fp64vec2,  0, y+(i+6 ));
        vec_xst(y4_fp64vec2,  0, y+(i+8 ));
        vec_xst(y5_fp64vec2,  0, y+(i+10));
        vec_xst(y6_fp64vec2,  0, y+(i+12));
        vec_xst(y7_fp64vec2,  0, y+(i+14));
        vec_xst(y8_fp64vec2,  0, y+(i+16));
        vec_xst(y9_fp64vec2,  0, y+(i+18));
        vec_xst(y10_fp64vec2, 0, y+(i+20));
        vec_xst(y11_fp64vec2, 0, y+(i+22));
    }
    for (; i <= n-8; i += 8)
    {
        x0_fp64vec2  = vec_xl(0, x+(i   ));
        x1_fp64vec2  = vec_xl(0, x+(i+2 ));
        x2_fp64vec2  = vec_xl(0, x+(i+4 ));
        x3_fp64vec2  = vec_xl(0, x+(i+6 ));

        y0_fp64vec2  = vec_div(x0_fp64vec2,  c_fp64vec2);
        y1_fp64vec2  = vec_div(x1_fp64vec2,  c_fp64vec2);
        y2_fp64vec2  = vec_div(x2_fp64vec2,  c_fp64vec2);
        y3_fp64vec2  = vec_div(x3_fp64vec2,  c_fp64vec2);

        vec_xst(y0_fp64vec2,  0, y+(i   ));
        vec_xst(y1_fp64vec2,  0, y+(i+2 ));
        vec_xst(y2_fp64vec2,  0, y+(i+4 ));
        vec_xst(y3_fp64vec2,  0, y+(i+6 ));

        vec_xst(y0_fp64vec2,  0, y+(i   ));
        vec_xst(y1_fp64vec2,  0, y+(i+2 ));
        vec_xst(y2_fp64vec2,  0, y+(i+4 ));
        vec_xst(y3_fp64vec2,  0, y+(i+6 ));
    }
    for (; i <= n-2; i += 2)
    {
        x0_fp64vec2  = vec_xl(0, x+(i   ));
        y0_fp64vec2  = vec_div(x0_fp64vec2,  c_fp64vec2);
        vec_xst(y0_fp64vec2,  0, y+(i   ));
    }
    for (; i < n; i++)
        y[i] = x[i] / c;
}


//--------------------------------------------------------------------------------------------------
// THFloatVector_fill_VSX:
//--------------------------------------------------------------------------------------------------
static void THFloatVector_fill_VSX(float *x, const float c, const ptrdiff_t n)
{
    ptrdiff_t i;

    float val[4] = {c, c, c, c};
    vector float fp32vec4 = vec_xl(0, val);

    for (i = 0; i <= n-256; i += 256)
    {
        vec_xst(fp32vec4, 0, x+(i    ));
        vec_xst(fp32vec4, 0, x+(i+4  ));
        vec_xst(fp32vec4, 0, x+(i+8  ));
        vec_xst(fp32vec4, 0, x+(i+12 ));
        vec_xst(fp32vec4, 0, x+(i+16 ));
        vec_xst(fp32vec4, 0, x+(i+20 ));
        vec_xst(fp32vec4, 0, x+(i+24 ));
        vec_xst(fp32vec4, 0, x+(i+28 ));
        vec_xst(fp32vec4, 0, x+(i+32 ));
        vec_xst(fp32vec4, 0, x+(i+36 ));
        vec_xst(fp32vec4, 0, x+(i+40 ));
        vec_xst(fp32vec4, 0, x+(i+44 ));
        vec_xst(fp32vec4, 0, x+(i+48 ));
        vec_xst(fp32vec4, 0, x+(i+52 ));
        vec_xst(fp32vec4, 0, x+(i+56 ));
        vec_xst(fp32vec4, 0, x+(i+60 ));
        vec_xst(fp32vec4, 0, x+(i+64 ));
        vec_xst(fp32vec4, 0, x+(i+68 ));
        vec_xst(fp32vec4, 0, x+(i+72 ));
        vec_xst(fp32vec4, 0, x+(i+76 ));
        vec_xst(fp32vec4, 0, x+(i+80 ));
        vec_xst(fp32vec4, 0, x+(i+84 ));
        vec_xst(fp32vec4, 0, x+(i+88 ));
        vec_xst(fp32vec4, 0, x+(i+92 ));
        vec_xst(fp32vec4, 0, x+(i+96 ));
        vec_xst(fp32vec4, 0, x+(i+100));
        vec_xst(fp32vec4, 0, x+(i+104));
        vec_xst(fp32vec4, 0, x+(i+108));
        vec_xst(fp32vec4, 0, x+(i+112));
        vec_xst(fp32vec4, 0, x+(i+116));
        vec_xst(fp32vec4, 0, x+(i+120));
        vec_xst(fp32vec4, 0, x+(i+124));
        vec_xst(fp32vec4, 0, x+(i+128));
        vec_xst(fp32vec4, 0, x+(i+132));
        vec_xst(fp32vec4, 0, x+(i+136));
        vec_xst(fp32vec4, 0, x+(i+140));
        vec_xst(fp32vec4, 0, x+(i+144));
        vec_xst(fp32vec4, 0, x+(i+148));
        vec_xst(fp32vec4, 0, x+(i+152));
        vec_xst(fp32vec4, 0, x+(i+156));
        vec_xst(fp32vec4, 0, x+(i+160));
        vec_xst(fp32vec4, 0, x+(i+164));
        vec_xst(fp32vec4, 0, x+(i+168));
        vec_xst(fp32vec4, 0, x+(i+172));
        vec_xst(fp32vec4, 0, x+(i+176));
        vec_xst(fp32vec4, 0, x+(i+180));
        vec_xst(fp32vec4, 0, x+(i+184));
        vec_xst(fp32vec4, 0, x+(i+188));
        vec_xst(fp32vec4, 0, x+(i+192));
        vec_xst(fp32vec4, 0, x+(i+196));
        vec_xst(fp32vec4, 0, x+(i+200));
        vec_xst(fp32vec4, 0, x+(i+204));
        vec_xst(fp32vec4, 0, x+(i+208));
        vec_xst(fp32vec4, 0, x+(i+212));
        vec_xst(fp32vec4, 0, x+(i+216));
        vec_xst(fp32vec4, 0, x+(i+220));
        vec_xst(fp32vec4, 0, x+(i+224));
        vec_xst(fp32vec4, 0, x+(i+228));
        vec_xst(fp32vec4, 0, x+(i+232));
        vec_xst(fp32vec4, 0, x+(i+236));
        vec_xst(fp32vec4, 0, x+(i+240));
        vec_xst(fp32vec4, 0, x+(i+244));
        vec_xst(fp32vec4, 0, x+(i+248));
        vec_xst(fp32vec4, 0, x+(i+252));
    }
    for (; i <= n-32; i += 32)
    {
        vec_xst(fp32vec4, 0, x+(i    ));
        vec_xst(fp32vec4, 0, x+(i+4  ));
        vec_xst(fp32vec4, 0, x+(i+8  ));
        vec_xst(fp32vec4, 0, x+(i+12 ));
        vec_xst(fp32vec4, 0, x+(i+16 ));
        vec_xst(fp32vec4, 0, x+(i+20 ));
        vec_xst(fp32vec4, 0, x+(i+24 ));
        vec_xst(fp32vec4, 0, x+(i+28 ));
    }
    for (; i <= n-4; i += 4)
        vec_xst(fp32vec4, 0, x+(i    ));
    for (; i < n; i++)
        x[i] = c;
}


//--------------------------------------------------------------------------------------------------
// THFloatVector_cadd_VSX:
//--------------------------------------------------------------------------------------------------
static void THFloatVector_cadd_VSX(float *z, const float *x, const float *y, const float c, const ptrdiff_t n)
{
    ptrdiff_t i;

    float val[4] = {c, c, c, c};
    vector float c_fp32vec4 = vec_xl(0, val);

    vector float y0_fp32vec4, y1_fp32vec4, y2_fp32vec4, y3_fp32vec4, y4_fp32vec4, y5_fp32vec4, y6_fp32vec4, y7_fp32vec4;
    vector float y8_fp32vec4, y9_fp32vec4, y10_fp32vec4, y11_fp32vec4;
    vector float x0_fp32vec4, x1_fp32vec4, x2_fp32vec4, x3_fp32vec4, x4_fp32vec4, x5_fp32vec4, x6_fp32vec4, x7_fp32vec4;
    vector float x8_fp32vec4, x9_fp32vec4, x10_fp32vec4, x11_fp32vec4;


    for (i = 0; i <= n-48; i += 48)
    {
        y0_fp32vec4  = vec_xl(0, y+(i   ));
        y1_fp32vec4  = vec_xl(0, y+(i+4 ));
        y2_fp32vec4  = vec_xl(0, y+(i+8 ));
        y3_fp32vec4  = vec_xl(0, y+(i+12));
        y4_fp32vec4  = vec_xl(0, y+(i+16 ));
        y5_fp32vec4  = vec_xl(0, y+(i+20));
        y6_fp32vec4  = vec_xl(0, y+(i+24));
        y7_fp32vec4  = vec_xl(0, y+(i+28));
        y8_fp32vec4  = vec_xl(0, y+(i+32));
        y9_fp32vec4  = vec_xl(0, y+(i+36));
        y10_fp32vec4 = vec_xl(0, y+(i+40));
        y11_fp32vec4 = vec_xl(0, y+(i+44));

        x0_fp32vec4  = vec_xl(0, x+(i   ));
        x1_fp32vec4  = vec_xl(0, x+(i+4 ));
        x2_fp32vec4  = vec_xl(0, x+(i+8 ));
        x3_fp32vec4  = vec_xl(0, x+(i+12 ));
        x4_fp32vec4  = vec_xl(0, x+(i+16 ));
        x5_fp32vec4  = vec_xl(0, x+(i+20));
        x6_fp32vec4  = vec_xl(0, x+(i+24));
        x7_fp32vec4  = vec_xl(0, x+(i+28));
        x8_fp32vec4  = vec_xl(0, x+(i+32));
        x9_fp32vec4  = vec_xl(0, x+(i+36));
        x10_fp32vec4 = vec_xl(0, x+(i+40));
        x11_fp32vec4 = vec_xl(0, x+(i+44));

        y0_fp32vec4  = vec_madd(y0_fp32vec4, c_fp32vec4,  x0_fp32vec4);
        y1_fp32vec4  = vec_madd(y1_fp32vec4, c_fp32vec4, x1_fp32vec4);
        y2_fp32vec4  = vec_madd(y2_fp32vec4, c_fp32vec4, x2_fp32vec4);
        y3_fp32vec4  = vec_madd(y3_fp32vec4, c_fp32vec4, x3_fp32vec4);
        y4_fp32vec4  = vec_madd(y4_fp32vec4, c_fp32vec4, x4_fp32vec4);
        y5_fp32vec4  = vec_madd(y5_fp32vec4, c_fp32vec4, x5_fp32vec4);
        y6_fp32vec4  = vec_madd(y6_fp32vec4, c_fp32vec4, x6_fp32vec4);
        y7_fp32vec4  = vec_madd(y7_fp32vec4, c_fp32vec4, x7_fp32vec4);
        y8_fp32vec4  = vec_madd(y8_fp32vec4, c_fp32vec4, x8_fp32vec4);
        y9_fp32vec4  = vec_madd(y9_fp32vec4, c_fp32vec4, x9_fp32vec4);
        y10_fp32vec4 = vec_madd(y10_fp32vec4, c_fp32vec4, x10_fp32vec4);
        y11_fp32vec4 = vec_madd(y11_fp32vec4, c_fp32vec4, x11_fp32vec4);

        vec_xst(y0_fp32vec4,  0, z+(i   ));
        vec_xst(y1_fp32vec4,  0, z+(i+4 ));
        vec_xst(y2_fp32vec4,  0, z+(i+8 ));
        vec_xst(y3_fp32vec4,  0, z+(i+12 ));
        vec_xst(y4_fp32vec4,  0, z+(i+16 ));
        vec_xst(y5_fp32vec4,  0, z+(i+20));
        vec_xst(y6_fp32vec4,  0, z+(i+24));
        vec_xst(y7_fp32vec4,  0, z+(i+28));
        vec_xst(y8_fp32vec4,  0, z+(i+32));
        vec_xst(y9_fp32vec4,  0, z+(i+36));
        vec_xst(y10_fp32vec4, 0, z+(i+40));
        vec_xst(y11_fp32vec4, 0, z+(i+44));
    }
    for (; i <= n-16; i += 16)
    {
        y0_fp32vec4  = vec_xl(0, y+(i   ));
        y1_fp32vec4  = vec_xl(0, y+(i+4 ));
        y2_fp32vec4  = vec_xl(0, y+(i+8 ));
        y3_fp32vec4  = vec_xl(0, y+(i+12 ));

        x0_fp32vec4  = vec_xl(0, x+(i   ));
        x1_fp32vec4  = vec_xl(0, x+(i+4 ));
        x2_fp32vec4  = vec_xl(0, x+(i+8 ));
        x3_fp32vec4  = vec_xl(0, x+(i+12 ));

        y0_fp32vec4  = vec_madd(y0_fp32vec4, c_fp32vec4, x0_fp32vec4);
        y1_fp32vec4  = vec_madd(y1_fp32vec4, c_fp32vec4, x1_fp32vec4);
        y2_fp32vec4  = vec_madd(y2_fp32vec4, c_fp32vec4, x2_fp32vec4);
        y3_fp32vec4  = vec_madd(y3_fp32vec4, c_fp32vec4, x3_fp32vec4);

        vec_xst(y0_fp32vec4,  0, z+(i   ));
        vec_xst(y1_fp32vec4,  0, z+(i+4 ));
        vec_xst(y2_fp32vec4,  0, z+(i+8 ));
        vec_xst(y3_fp32vec4,  0, z+(i+12 ));
    }
    for (; i <= n-4; i += 4)
    {
        y0_fp32vec4  = vec_xl(0, y+(i   ));
        x0_fp32vec4  = vec_xl(0, x+(i   ));
        y0_fp32vec4  = vec_madd(y0_fp32vec4, c_fp32vec4, x0_fp32vec4);
        vec_xst(y0_fp32vec4,  0, z+(i   ));
    }
    for (; i < n; i++)
        z[i] = x[i] + c* y[i];
}


//--------------------------------------------------------------------------------------------------
// THFloatVector_adds_VSX:
//--------------------------------------------------------------------------------------------------
static void THFloatVector_adds_VSX(float *y, const float *x, const float c, const ptrdiff_t n)
{
    ptrdiff_t i;
    float val[4] = {c, c, c, c};
    vector float c_fp32vec4 = vec_xl(0, val);

    vector float y0_fp32vec4, y1_fp32vec4, y2_fp32vec4, y3_fp32vec4, y4_fp32vec4, y5_fp32vec4, y6_fp32vec4, y7_fp32vec4;
    vector float y8_fp32vec4, y9_fp32vec4, y10_fp32vec4, y11_fp32vec4;
    vector float x0_fp32vec4, x1_fp32vec4, x2_fp32vec4, x3_fp32vec4, x4_fp32vec4, x5_fp32vec4, x6_fp32vec4, x7_fp32vec4;
    vector float x8_fp32vec4, x9_fp32vec4, x10_fp32vec4, x11_fp32vec4;


    for (i = 0; i <= n-48; i += 48)
    {
        x0_fp32vec4  = vec_xl(0, x+(i   ));
        x1_fp32vec4  = vec_xl(0, x+(i+4 ));
        x2_fp32vec4  = vec_xl(0, x+(i+8 ));
        x3_fp32vec4  = vec_xl(0, x+(i+12));
        x4_fp32vec4  = vec_xl(0, x+(i+16));
        x5_fp32vec4  = vec_xl(0, x+(i+20));
        x6_fp32vec4  = vec_xl(0, x+(i+24));
        x7_fp32vec4  = vec_xl(0, x+(i+28));
        x8_fp32vec4  = vec_xl(0, x+(i+32));
        x9_fp32vec4  = vec_xl(0, x+(i+36));
        x10_fp32vec4 = vec_xl(0, x+(i+40));
        x11_fp32vec4 = vec_xl(0, x+(i+44));

        y0_fp32vec4  = vec_add(x0_fp32vec4,  c_fp32vec4);
        y1_fp32vec4  = vec_add(x1_fp32vec4,  c_fp32vec4);
        y2_fp32vec4  = vec_add(x2_fp32vec4,  c_fp32vec4);
        y3_fp32vec4  = vec_add(x3_fp32vec4,  c_fp32vec4);
        y4_fp32vec4  = vec_add(x4_fp32vec4,  c_fp32vec4);
        y5_fp32vec4  = vec_add(x5_fp32vec4,  c_fp32vec4);
        y6_fp32vec4  = vec_add(x6_fp32vec4,  c_fp32vec4);
        y7_fp32vec4  = vec_add(x7_fp32vec4,  c_fp32vec4);
        y8_fp32vec4  = vec_add(x8_fp32vec4,  c_fp32vec4);
        y9_fp32vec4  = vec_add(x9_fp32vec4,  c_fp32vec4);
        y10_fp32vec4 = vec_add(x10_fp32vec4, c_fp32vec4);
        y11_fp32vec4 = vec_add(x11_fp32vec4, c_fp32vec4);

        vec_xst(y0_fp32vec4,  0, y+(i   ));
        vec_xst(y1_fp32vec4,  0, y+(i+4 ));
        vec_xst(y2_fp32vec4,  0, y+(i+8 ));
        vec_xst(y3_fp32vec4,  0, y+(i+12));
        vec_xst(y4_fp32vec4,  0, y+(i+16));
        vec_xst(y5_fp32vec4,  0, y+(i+20));
        vec_xst(y6_fp32vec4,  0, y+(i+24));
        vec_xst(y7_fp32vec4,  0, y+(i+28));
        vec_xst(y8_fp32vec4,  0, y+(i+32));
        vec_xst(y9_fp32vec4,  0, y+(i+36));
        vec_xst(y10_fp32vec4, 0, y+(i+40));
        vec_xst(y11_fp32vec4, 0, y+(i+44));
    }
    for (; i <= n-16; i += 16)
    {
        x0_fp32vec4  = vec_xl(0, x+(i   ));
        x1_fp32vec4  = vec_xl(0, x+(i+4 ));
        x2_fp32vec4  = vec_xl(0, x+(i+8 ));
        x3_fp32vec4  = vec_xl(0, x+(i+12));

        y0_fp32vec4  = vec_add(x0_fp32vec4,  c_fp32vec4);
        y1_fp32vec4  = vec_add(x1_fp32vec4,  c_fp32vec4);
        y2_fp32vec4  = vec_add(x2_fp32vec4,  c_fp32vec4);
        y3_fp32vec4  = vec_add(x3_fp32vec4,  c_fp32vec4);

        vec_xst(y0_fp32vec4,  0, y+(i   ));
        vec_xst(y1_fp32vec4,  0, y+(i+4 ));
        vec_xst(y2_fp32vec4,  0, y+(i+8 ));
        vec_xst(y3_fp32vec4,  0, y+(i+12));
    }
    for (; i <= n-4; i += 4)
    {
        x0_fp32vec4  = vec_xl(0, x+(i   ));
        y0_fp32vec4  = vec_add(x0_fp32vec4,  c_fp32vec4);
        vec_xst(y0_fp32vec4,  0, y+(i   ));
    }
    for (; i < n; i++)
        y[i] = c + x[i];
}


//--------------------------------------------------------------------------------------------------
// THFloatVector_cmul_VSX:
//--------------------------------------------------------------------------------------------------
static void THFloatVector_cmul_VSX(float *z, const float *y, const float *x, const ptrdiff_t n)
{
    ptrdiff_t i;

    vector float y0_fp32vec4, y1_fp32vec4, y2_fp32vec4, y3_fp32vec4, y4_fp32vec4, y5_fp32vec4, y6_fp32vec4, y7_fp32vec4;
    vector float y8_fp32vec4, y9_fp32vec4, y10_fp32vec4, y11_fp32vec4;
    vector float x0_fp32vec4, x1_fp32vec4, x2_fp32vec4, x3_fp32vec4, x4_fp32vec4, x5_fp32vec4, x6_fp32vec4, x7_fp32vec4;
    vector float x8_fp32vec4, x9_fp32vec4, x10_fp32vec4, x11_fp32vec4;


    for (i = 0; i <= n-48; i += 48)
    {
        y0_fp32vec4  = vec_xl(0, y+(i   ));
        y1_fp32vec4  = vec_xl(0, y+(i+4 ));
        y2_fp32vec4  = vec_xl(0, y+(i+8 ));
        y3_fp32vec4  = vec_xl(0, y+(i+12 ));
        y4_fp32vec4  = vec_xl(0, y+(i+16 ));
        y5_fp32vec4  = vec_xl(0, y+(i+20));
        y6_fp32vec4  = vec_xl(0, y+(i+24));
        y7_fp32vec4  = vec_xl(0, y+(i+28));
        y8_fp32vec4  = vec_xl(0, y+(i+32));
        y9_fp32vec4  = vec_xl(0, y+(i+36));
        y10_fp32vec4 = vec_xl(0, y+(i+40));
        y11_fp32vec4 = vec_xl(0, y+(i+44));

        x0_fp32vec4  = vec_xl(0, x+(i   ));
        x1_fp32vec4  = vec_xl(0, x+(i+4 ));
        x2_fp32vec4  = vec_xl(0, x+(i+8 ));
        x3_fp32vec4  = vec_xl(0, x+(i+12 ));
        x4_fp32vec4  = vec_xl(0, x+(i+16 ));
        x5_fp32vec4  = vec_xl(0, x+(i+20));
        x6_fp32vec4  = vec_xl(0, x+(i+24));
        x7_fp32vec4  = vec_xl(0, x+(i+28));
        x8_fp32vec4  = vec_xl(0, x+(i+32));
        x9_fp32vec4  = vec_xl(0, x+(i+36));
        x10_fp32vec4 = vec_xl(0, x+(i+40));
        x11_fp32vec4 = vec_xl(0, x+(i+44));

        y0_fp32vec4  = vec_mul(y0_fp32vec4,  x0_fp32vec4);
        y1_fp32vec4  = vec_mul(y1_fp32vec4,  x1_fp32vec4);
        y2_fp32vec4  = vec_mul(y2_fp32vec4,  x2_fp32vec4);
        y3_fp32vec4  = vec_mul(y3_fp32vec4,  x3_fp32vec4);
        y4_fp32vec4  = vec_mul(y4_fp32vec4,  x4_fp32vec4);
        y5_fp32vec4  = vec_mul(y5_fp32vec4,  x5_fp32vec4);
        y6_fp32vec4  = vec_mul(y6_fp32vec4,  x6_fp32vec4);
        y7_fp32vec4  = vec_mul(y7_fp32vec4,  x7_fp32vec4);
        y8_fp32vec4  = vec_mul(y8_fp32vec4,  x8_fp32vec4);
        y9_fp32vec4  = vec_mul(y9_fp32vec4,  x9_fp32vec4);
        y10_fp32vec4 = vec_mul(y10_fp32vec4, x10_fp32vec4);
        y11_fp32vec4 = vec_mul(y11_fp32vec4, x11_fp32vec4);

        vec_xst(y0_fp32vec4,  0, z+(i   ));
        vec_xst(y1_fp32vec4,  0, z+(i+4 ));
        vec_xst(y2_fp32vec4,  0, z+(i+8 ));
        vec_xst(y3_fp32vec4,  0, z+(i+12 ));
        vec_xst(y4_fp32vec4,  0, z+(i+16 ));
        vec_xst(y5_fp32vec4,  0, z+(i+20));
        vec_xst(y6_fp32vec4,  0, z+(i+24));
        vec_xst(y7_fp32vec4,  0, z+(i+28));
        vec_xst(y8_fp32vec4,  0, z+(i+32));
        vec_xst(y9_fp32vec4,  0, z+(i+36));
        vec_xst(y10_fp32vec4, 0, z+(i+40));
        vec_xst(y11_fp32vec4, 0, z+(i+44));
    }
    for (; i <= n-16; i += 16)
    {
        y0_fp32vec4  = vec_xl(0, y+(i   ));
        y1_fp32vec4  = vec_xl(0, y+(i+4 ));
        y2_fp32vec4  = vec_xl(0, y+(i+8 ));
        y3_fp32vec4  = vec_xl(0, y+(i+12 ));

        x0_fp32vec4  = vec_xl(0, x+(i   ));
        x1_fp32vec4  = vec_xl(0, x+(i+4 ));
        x2_fp32vec4  = vec_xl(0, x+(i+8 ));
        x3_fp32vec4  = vec_xl(0, x+(i+12 ));

        y0_fp32vec4  = vec_mul(y0_fp32vec4,  x0_fp32vec4);
        y1_fp32vec4  = vec_mul(y1_fp32vec4,  x1_fp32vec4);
        y2_fp32vec4  = vec_mul(y2_fp32vec4,  x2_fp32vec4);
        y3_fp32vec4  = vec_mul(y3_fp32vec4,  x3_fp32vec4);

        vec_xst(y0_fp32vec4,  0, z+(i   ));
        vec_xst(y1_fp32vec4,  0, z+(i+4 ));
        vec_xst(y2_fp32vec4,  0, z+(i+8 ));
        vec_xst(y3_fp32vec4,  0, z+(i+12 ));
    }
    for (; i <= n-4; i += 4)
    {
        y0_fp32vec4  = vec_xl(0, y+(i   ));
        x0_fp32vec4  = vec_xl(0, x+(i   ));
        y0_fp32vec4  = vec_mul(y0_fp32vec4,  x0_fp32vec4);
        vec_xst(y0_fp32vec4,  0, z+(i   ));
    }
    for (; i < n; i++)
        z[i] = y[i] * x[i];
}


//--------------------------------------------------------------------------------------------------
// THFloatVector_muls_VSX:
//--------------------------------------------------------------------------------------------------
static void THFloatVector_muls_VSX(float *y, const float *x, const float c, const ptrdiff_t n)
{
    ptrdiff_t i;
    float val[4] = {c, c, c, c};
    vector float c_fp32vec4 = vec_xl(0, val);

    vector float y0_fp32vec4, y1_fp32vec4, y2_fp32vec4, y3_fp32vec4, y4_fp32vec4, y5_fp32vec4, y6_fp32vec4, y7_fp32vec4;
    vector float y8_fp32vec4, y9_fp32vec4, y10_fp32vec4, y11_fp32vec4;
    vector float x0_fp32vec4, x1_fp32vec4, x2_fp32vec4, x3_fp32vec4, x4_fp32vec4, x5_fp32vec4, x6_fp32vec4, x7_fp32vec4;
    vector float x8_fp32vec4, x9_fp32vec4, x10_fp32vec4, x11_fp32vec4;


    for (i = 0; i <= n-48; i += 48)
    {
        x0_fp32vec4  = vec_xl(0, x+(i   ));
        x1_fp32vec4  = vec_xl(0, x+(i+4 ));
        x2_fp32vec4  = vec_xl(0, x+(i+8 ));
        x3_fp32vec4  = vec_xl(0, x+(i+12));
        x4_fp32vec4  = vec_xl(0, x+(i+16));
        x5_fp32vec4  = vec_xl(0, x+(i+20));
        x6_fp32vec4  = vec_xl(0, x+(i+24));
        x7_fp32vec4  = vec_xl(0, x+(i+28));
        x8_fp32vec4  = vec_xl(0, x+(i+32));
        x9_fp32vec4  = vec_xl(0, x+(i+36));
        x10_fp32vec4 = vec_xl(0, x+(i+40));
        x11_fp32vec4 = vec_xl(0, x+(i+44));

        y0_fp32vec4  = vec_mul(x0_fp32vec4,  c_fp32vec4);
        y1_fp32vec4  = vec_mul(x1_fp32vec4,  c_fp32vec4);
        y2_fp32vec4  = vec_mul(x2_fp32vec4,  c_fp32vec4);
        y3_fp32vec4  = vec_mul(x3_fp32vec4,  c_fp32vec4);
        y4_fp32vec4  = vec_mul(x4_fp32vec4,  c_fp32vec4);
        y5_fp32vec4  = vec_mul(x5_fp32vec4,  c_fp32vec4);
        y6_fp32vec4  = vec_mul(x6_fp32vec4,  c_fp32vec4);
        y7_fp32vec4  = vec_mul(x7_fp32vec4,  c_fp32vec4);
        y8_fp32vec4  = vec_mul(x8_fp32vec4,  c_fp32vec4);
        y9_fp32vec4  = vec_mul(x9_fp32vec4,  c_fp32vec4);
        y10_fp32vec4 = vec_mul(x10_fp32vec4, c_fp32vec4);
        y11_fp32vec4 = vec_mul(x11_fp32vec4, c_fp32vec4);

        vec_xst(y0_fp32vec4,  0, y+(i   ));
        vec_xst(y1_fp32vec4,  0, y+(i+4 ));
        vec_xst(y2_fp32vec4,  0, y+(i+8 ));
        vec_xst(y3_fp32vec4,  0, y+(i+12));
        vec_xst(y4_fp32vec4,  0, y+(i+16));
        vec_xst(y5_fp32vec4,  0, y+(i+20));
        vec_xst(y6_fp32vec4,  0, y+(i+24));
        vec_xst(y7_fp32vec4,  0, y+(i+28));
        vec_xst(y8_fp32vec4,  0, y+(i+32));
        vec_xst(y9_fp32vec4,  0, y+(i+36));
        vec_xst(y10_fp32vec4, 0, y+(i+40));
        vec_xst(y11_fp32vec4, 0, y+(i+44));
    }
    for (; i <= n-16; i += 16)
    {
        x0_fp32vec4  = vec_xl(0, x+(i   ));
        x1_fp32vec4  = vec_xl(0, x+(i+4 ));
        x2_fp32vec4  = vec_xl(0, x+(i+8 ));
        x3_fp32vec4  = vec_xl(0, x+(i+12));

        y0_fp32vec4  = vec_mul(x0_fp32vec4,  c_fp32vec4);
        y1_fp32vec4  = vec_mul(x1_fp32vec4,  c_fp32vec4);
        y2_fp32vec4  = vec_mul(x2_fp32vec4,  c_fp32vec4);
        y3_fp32vec4  = vec_mul(x3_fp32vec4,  c_fp32vec4);

        vec_xst(y0_fp32vec4,  0, y+(i   ));
        vec_xst(y1_fp32vec4,  0, y+(i+4 ));
        vec_xst(y2_fp32vec4,  0, y+(i+8 ));
        vec_xst(y3_fp32vec4,  0, y+(i+12));
    }
    for (; i <= n-4; i += 4)
    {
        x0_fp32vec4  = vec_xl(0, x+(i   ));
        y0_fp32vec4  = vec_mul(x0_fp32vec4,  c_fp32vec4);
        vec_xst(y0_fp32vec4,  0, y+(i   ));
    }
    for (; i < n; i++)
        y[i] = c * x[i];
}


//--------------------------------------------------------------------------------------------------
// THFloatVector_cdiv_VSX:
//--------------------------------------------------------------------------------------------------
static void THFloatVector_cdiv_VSX(float *z, const float *x, const float *y, const ptrdiff_t n)
{
    ptrdiff_t i;

    vector float y0_fp32vec4, y1_fp32vec4, y2_fp32vec4, y3_fp32vec4, y4_fp32vec4, y5_fp32vec4, y6_fp32vec4, y7_fp32vec4;
    vector float y8_fp32vec4, y9_fp32vec4, y10_fp32vec4, y11_fp32vec4;
    vector float x0_fp32vec4, x1_fp32vec4, x2_fp32vec4, x3_fp32vec4, x4_fp32vec4, x5_fp32vec4, x6_fp32vec4, x7_fp32vec4;
    vector float x8_fp32vec4, x9_fp32vec4, x10_fp32vec4, x11_fp32vec4;


    for (i = 0; i <= n-48; i += 48)
    {
        y0_fp32vec4  = vec_xl(0, y+(i   ));
        y1_fp32vec4  = vec_xl(0, y+(i+4));
        y2_fp32vec4  = vec_xl(0, y+(i+8));
        y3_fp32vec4  = vec_xl(0, y+(i+12));
        y4_fp32vec4  = vec_xl(0, y+(i+16));
        y5_fp32vec4  = vec_xl(0, y+(i+20));
        y6_fp32vec4  = vec_xl(0, y+(i+24));
        y7_fp32vec4  = vec_xl(0, y+(i+28));
        y8_fp32vec4  = vec_xl(0, y+(i+32));
        y9_fp32vec4  = vec_xl(0, y+(i+36));
        y10_fp32vec4 = vec_xl(0, y+(i+40));
        y11_fp32vec4 = vec_xl(0, y+(i+44));

        x0_fp32vec4  = vec_xl(0, x+(i   ));
        x1_fp32vec4  = vec_xl(0, x+(i+4 ));
        x2_fp32vec4  = vec_xl(0, x+(i+8 ));
        x3_fp32vec4  = vec_xl(0, x+(i+12 ));
        x4_fp32vec4  = vec_xl(0, x+(i+16 ));
        x5_fp32vec4  = vec_xl(0, x+(i+20));
        x6_fp32vec4  = vec_xl(0, x+(i+24));
        x7_fp32vec4  = vec_xl(0, x+(i+28));
        x8_fp32vec4  = vec_xl(0, x+(i+32));
        x9_fp32vec4  = vec_xl(0, x+(i+36));
        x10_fp32vec4 = vec_xl(0, x+(i+40));
        x11_fp32vec4 = vec_xl(0, x+(i+44));

        y0_fp32vec4  = vec_div(x0_fp32vec4,  y0_fp32vec4);
        y1_fp32vec4  = vec_div(x1_fp32vec4,  y1_fp32vec4);
        y2_fp32vec4  = vec_div(x2_fp32vec4,  y2_fp32vec4);
        y3_fp32vec4  = vec_div(x3_fp32vec4,  y3_fp32vec4);
        y4_fp32vec4  = vec_div(x4_fp32vec4,  y4_fp32vec4);
        y5_fp32vec4  = vec_div(x5_fp32vec4,  y5_fp32vec4);
        y6_fp32vec4  = vec_div(x6_fp32vec4,  y6_fp32vec4);
        y7_fp32vec4  = vec_div(x7_fp32vec4,  y7_fp32vec4);
        y8_fp32vec4  = vec_div(x8_fp32vec4,  y8_fp32vec4);
        y9_fp32vec4  = vec_div(x9_fp32vec4,  y9_fp32vec4);
        y10_fp32vec4 = vec_div(x10_fp32vec4, y10_fp32vec4);
        y11_fp32vec4 = vec_div(x11_fp32vec4, y11_fp32vec4);

        vec_xst(y0_fp32vec4,  0, z+(i   ));
        vec_xst(y1_fp32vec4,  0, z+(i+4 ));
        vec_xst(y2_fp32vec4,  0, z+(i+8 ));
        vec_xst(y3_fp32vec4,  0, z+(i+12 ));
        vec_xst(y4_fp32vec4,  0, z+(i+16 ));
        vec_xst(y5_fp32vec4,  0, z+(i+20));
        vec_xst(y6_fp32vec4,  0, z+(i+24));
        vec_xst(y7_fp32vec4,  0, z+(i+28));
        vec_xst(y8_fp32vec4,  0, z+(i+32));
        vec_xst(y9_fp32vec4,  0, z+(i+36));
        vec_xst(y10_fp32vec4, 0, z+(i+40));
        vec_xst(y11_fp32vec4, 0, z+(i+44));
    }
    for (; i <= n-16; i += 16)
    {
        y0_fp32vec4  = vec_xl(0, y+(i   ));
        y1_fp32vec4  = vec_xl(0, y+(i+4 ));
        y2_fp32vec4  = vec_xl(0, y+(i+8 ));
        y3_fp32vec4  = vec_xl(0, y+(i+12 ));

        x0_fp32vec4  = vec_xl(0, x+(i   ));
        x1_fp32vec4  = vec_xl(0, x+(i+4 ));
        x2_fp32vec4  = vec_xl(0, x+(i+8 ));
        x3_fp32vec4  = vec_xl(0, x+(i+12 ));

        y0_fp32vec4  = vec_div(x0_fp32vec4,  y0_fp32vec4);
        y1_fp32vec4  = vec_div(x1_fp32vec4,  y1_fp32vec4);
        y2_fp32vec4  = vec_div(x2_fp32vec4,  y2_fp32vec4);
        y3_fp32vec4  = vec_div(x3_fp32vec4,  y3_fp32vec4);

        vec_xst(y0_fp32vec4,  0, z+(i   ));
        vec_xst(y1_fp32vec4,  0, z+(i+4 ));
        vec_xst(y2_fp32vec4,  0, z+(i+8 ));
        vec_xst(y3_fp32vec4,  0, z+(i+12 ));
    }
    for (; i <= n-4; i += 4)
    {
        y0_fp32vec4  = vec_xl(0, y+(i   ));
        x0_fp32vec4  = vec_xl(0, x+(i   ));
        y0_fp32vec4  = vec_div(x0_fp32vec4,  y0_fp32vec4);
        vec_xst(y0_fp32vec4,  0, z+(i   ));
    }
    for (; i < n; i++)
        z[i] = x[i] / y[i];
}


//--------------------------------------------------------------------------------------------------
// THFloatVector_divs_VSX:
//--------------------------------------------------------------------------------------------------
static void THFloatVector_divs_VSX(float *y, const float*x, const float c, const ptrdiff_t n)
{
    ptrdiff_t i;

    float val[4] = {c, c, c, c};
    vector float c_fp64vec2 = vec_xl(0, val);

    vector float y0_fp64vec2, y1_fp64vec2, y2_fp64vec2, y3_fp64vec2, y4_fp64vec2, y5_fp64vec2, y6_fp64vec2, y7_fp64vec2;
    vector float y8_fp64vec2, y9_fp64vec2, y10_fp64vec2, y11_fp64vec2;
    vector float x0_fp64vec2, x1_fp64vec2, x2_fp64vec2, x3_fp64vec2, x4_fp64vec2, x5_fp64vec2, x6_fp64vec2, x7_fp64vec2;
    vector float x8_fp64vec2, x9_fp64vec2, x10_fp64vec2, x11_fp64vec2;


    for (i = 0; i <= n-48; i += 48)
    {
        x0_fp64vec2  = vec_xl(0, x+(i   ));
        x1_fp64vec2  = vec_xl(0, x+(i+4 ));
        x2_fp64vec2  = vec_xl(0, x+(i+8 ));
        x3_fp64vec2  = vec_xl(0, x+(i+12 ));
        x4_fp64vec2  = vec_xl(0, x+(i+16 ));
        x5_fp64vec2  = vec_xl(0, x+(i+20));
        x6_fp64vec2  = vec_xl(0, x+(i+24));
        x7_fp64vec2  = vec_xl(0, x+(i+28));
        x8_fp64vec2  = vec_xl(0, x+(i+32));
        x9_fp64vec2  = vec_xl(0, x+(i+36));
        x10_fp64vec2 = vec_xl(0, x+(i+40));
        x11_fp64vec2 = vec_xl(0, x+(i+44));

        y0_fp64vec2  = vec_div(x0_fp64vec2,  c_fp64vec2);
        y1_fp64vec2  = vec_div(x1_fp64vec2,  c_fp64vec2);
        y2_fp64vec2  = vec_div(x2_fp64vec2,  c_fp64vec2);
        y3_fp64vec2  = vec_div(x3_fp64vec2,  c_fp64vec2);
        y4_fp64vec2  = vec_div(x4_fp64vec2,  c_fp64vec2);
        y5_fp64vec2  = vec_div(x5_fp64vec2,  c_fp64vec2);
        y6_fp64vec2  = vec_div(x6_fp64vec2,  c_fp64vec2);
        y7_fp64vec2  = vec_div(x7_fp64vec2,  c_fp64vec2);
        y8_fp64vec2  = vec_div(x8_fp64vec2,  c_fp64vec2);
        y9_fp64vec2  = vec_div(x9_fp64vec2,  c_fp64vec2);
        y10_fp64vec2 = vec_div(x10_fp64vec2, c_fp64vec2);
        y11_fp64vec2 = vec_div(x11_fp64vec2, c_fp64vec2);
        

        vec_xst(y0_fp64vec2,  0, y+(i   ));
        vec_xst(y1_fp64vec2,  0, y+(i+4 ));
        vec_xst(y2_fp64vec2,  0, y+(i+8 ));
        vec_xst(y3_fp64vec2,  0, y+(i+12 ));
        vec_xst(y4_fp64vec2,  0, y+(i+16 ));
        vec_xst(y5_fp64vec2,  0, y+(i+20));
        vec_xst(y6_fp64vec2,  0, y+(i+24));
        vec_xst(y7_fp64vec2,  0, y+(i+28));
        vec_xst(y8_fp64vec2,  0, y+(i+32));
        vec_xst(y9_fp64vec2,  0, y+(i+36));
        vec_xst(y10_fp64vec2, 0, y+(i+40));
        vec_xst(y11_fp64vec2, 0, y+(i+44));
    }
    for (; i <= n-16; i += 16)
    {
        x0_fp64vec2  = vec_xl(0, x+(i   ));
        x1_fp64vec2  = vec_xl(0, x+(i+4 ));
        x2_fp64vec2  = vec_xl(0, x+(i+8 ));
        x3_fp64vec2  = vec_xl(0, x+(i+12 ));

        y0_fp64vec2  = vec_div(x0_fp64vec2,  c_fp64vec2);
        y1_fp64vec2  = vec_div(x1_fp64vec2,  c_fp64vec2);
        y2_fp64vec2  = vec_div(x2_fp64vec2,  c_fp64vec2);
        y3_fp64vec2  = vec_div(x3_fp64vec2,  c_fp64vec2);

        vec_xst(y0_fp64vec2,  0, y+(i   ));
        vec_xst(y1_fp64vec2,  0, y+(i+4 ));
        vec_xst(y2_fp64vec2,  0, y+(i+8 ));
        vec_xst(y3_fp64vec2,  0, y+(i+12 ));

        vec_xst(y0_fp64vec2,  0, y+(i   ));
        vec_xst(y1_fp64vec2,  0, y+(i+4 ));
        vec_xst(y2_fp64vec2,  0, y+(i+8 ));
        vec_xst(y3_fp64vec2,  0, y+(i+16 ));
    }
    for (; i <= n-4; i += 4)
    {
        x0_fp64vec2  = vec_xl(0, x+(i   ));
        y0_fp64vec2  = vec_div(x0_fp64vec2,  c_fp64vec2);
        vec_xst(y0_fp64vec2,  0, y+(i   ));
    }
    for (; i < n; i++)
        y[i] = x[i] / c;
}


//------------------------------------------------
//
// Testing for correctness and performance
//
// If you want to run these tests, compile this
// file with -DRUN_VSX_TESTS on a Power machine,
// and then run the executable that is generated.
//
//------------------------------------------------
//
// Example passing run (from a Power8 machine):
//
//    $ gcc VSX.c -O2 -D RUN_VSX_TESTS -o vsxtest
//    $ ./vsxtest
//
//	TODO
//
//
//    Finished runnning all tests. All tests PASSED.
//
//------------------------------------------------
#ifdef RUN_VSX_TESTS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <math.h>

#define VSX_PERF_NUM_TEST_ELEMENTS 100000000
#define VSX_FUNC_NUM_TEST_ELEMENTS 2507


//--------------------------------------------------------------------------------------------------
// Standard implementations:
//--------------------------------------------------------------------------------------------------
static void standardDouble_fill(double *x, const double c, const ptrdiff_t n)
{
    for (ptrdiff_t i = 0; i < n; i++)
        x[i] = c;
}

static void standardFloat_fill(float *x, const float c, const ptrdiff_t n)
{
    for (ptrdiff_t i = 0; i < n; i++)
        x[i] = c;
}

static void standardDouble_cadd(double *z, const double *x,  const double *y, const double c, const ptrdiff_t n)
{
  for (ptrdiff_t i = 0; i < n; i++)
    z[i] = x[i] + c * y[i];
}

static void standardFloat_cadd(float *z, const float *x, const float *y, const float c, const ptrdiff_t n)
{
  for (ptrdiff_t i = 0; i < n; i++)
    z[i] = x[i] + c * y[i];
}

static void standardDouble_adds(double *y, const double *x, const double c, const ptrdiff_t n)
{
  for (ptrdiff_t i = 0; i < n; i++)
    y[i] = c + x[i];
}

static void standardFloat_adds(float *y, const float *x, const float c, const ptrdiff_t n)
{
  for (ptrdiff_t i = 0; i < n; i++)
    y[i] = c + x[i];
}

static void standardDouble_cmul(double *z, const double *x,  const double *y, const ptrdiff_t n)
{
  for (ptrdiff_t i = 0; i < n; i++)
    z[i] = x[i] * y[i];
}

static void standardFloat_cmul(float *z, const float *x, const float *y, const ptrdiff_t n)
{
  for (ptrdiff_t i = 0; i < n; i++)
    z[i] = x[i] * y[i];
}

static void standardDouble_muls(double *y, const double *x, const double c, const ptrdiff_t n)
{
  for (ptrdiff_t i = 0; i < n; i++)
    y[i] = c * x[i];
}

static void standardFloat_muls(float *y, const float *x, const float c, const ptrdiff_t n)
{
  for (ptrdiff_t i = 0; i < n; i++)
    y[i] = c * x[i];
}

static void standardDouble_cdiv(double *z, const double *x,  const double *y, const ptrdiff_t n)
{
  for (ptrdiff_t i = 0; i < n; i++)
    z[i] = x[i] / y[i];
}

static void standardFloat_cdiv(float *z, const float *x, const float *y, const ptrdiff_t n)
{
  for (ptrdiff_t i = 0; i < n; i++)
    z[i] = x[i] / y[i];
}

static void standardDouble_divs(double *y, const double *x, const double c, const ptrdiff_t n)
{
  for (ptrdiff_t i = 0; i < n; i++)
    y[i] = x[i] / c;
}

static void standardFloat_divs(float *y, const float *x, const float c, const ptrdiff_t n)
{
  for (ptrdiff_t i = 0; i < n; i++)
    y[i] = x[i] / c;
}

double randDouble()
{
    return (double)(rand()%100)/(double)(rand()%100) * (rand()%2 ? -1.0 : 1.0);
}

int near(double a, double b)
{
    int aClass = fpclassify(a);
    int bClass = fpclassify(b);

    if(aClass != bClass)             // i.e. is it NAN, infinite, or finite...?
        return 0;

    if(aClass == FP_INFINITE)       // if it is infinite, the sign must be the same, i.e. positive infinity is not near negative infinity
        return (signbit(a) == signbit(b));
    else if(aClass == FP_NORMAL)    // if it is a normal number then check the magnitude of the difference between the numbers
        return fabs(a - b) < 0.001;
    else                            // if both number are of the same class as each other and are of any other class (i.e. such as NAN), then they are near to each other.
        return 1;
}


//--------------------------------------------------------------------------------------------------
// Standard tests:
//--------------------------------------------------------------------------------------------------
void test_THDoubleVector_fill_VSX()
{
    clock_t start, end;
    double elapsedSeconds_optimized, elapsedSeconds_standard;

    double *x_standard  = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *x_optimized = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));

    double yVal0 = 17.2;
    double yVal1 = 8.2;
    double yVal2 = 5.1;
    double yVal3 = -0.9;

    //-------------------------------------------------
    // Performance Test
    //-------------------------------------------------
    start = clock();
    standardDouble_fill(x_standard, yVal0, VSX_PERF_NUM_TEST_ELEMENTS  );
    standardDouble_fill(x_standard, yVal1, VSX_PERF_NUM_TEST_ELEMENTS-1);
    standardDouble_fill(x_standard, yVal2, VSX_PERF_NUM_TEST_ELEMENTS-2);
    standardDouble_fill(x_standard, yVal3, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_standard = (double)(end - start) / CLOCKS_PER_SEC;
    printf("standardDouble_fill() test took %.5lf seconds\n", elapsedSeconds_standard);

    start = clock();
    THDoubleVector_fill_VSX(x_optimized, yVal0, VSX_PERF_NUM_TEST_ELEMENTS  );
    THDoubleVector_fill_VSX(x_optimized, yVal1, VSX_PERF_NUM_TEST_ELEMENTS-1);
    THDoubleVector_fill_VSX(x_optimized, yVal2, VSX_PERF_NUM_TEST_ELEMENTS-2);
    THDoubleVector_fill_VSX(x_optimized, yVal3, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_optimized = (double)(end - start) / CLOCKS_PER_SEC;
    printf("THDoubleVector_fill_VSX() test took %.5lf seconds\n", elapsedSeconds_optimized);


    //-------------------------------------------------
    // Correctness Test
    //-------------------------------------------------
    yVal0 += 1.0;
    yVal1 += 1.0;
    yVal2 += 1.0;
    yVal3 -= 1.0;

    standardDouble_fill(    x_standard,  yVal0, VSX_FUNC_NUM_TEST_ELEMENTS);
    THDoubleVector_fill_VSX(x_optimized, yVal0, VSX_FUNC_NUM_TEST_ELEMENTS);
    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
        assert(x_optimized[i] == yVal0);

    standardDouble_fill(    x_standard+1,  yVal1, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    THDoubleVector_fill_VSX(x_optimized+1, yVal1, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    standardDouble_fill(    x_standard+2,  yVal2, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    THDoubleVector_fill_VSX(x_optimized+2, yVal2, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    standardDouble_fill(    x_standard+3,  yVal3, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    THDoubleVector_fill_VSX(x_optimized+3, yVal3, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    standardDouble_fill(    x_standard+517,  yVal0, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    THDoubleVector_fill_VSX(x_optimized+517, yVal0, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    int r = rand() % 258;
    standardDouble_fill(    x_standard+517+r,  yVal2, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    THDoubleVector_fill_VSX(x_optimized+517+r, yVal2, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
        assert(x_optimized[i] == x_standard[i]);
    printf("All assertions PASSED for THDoubleVector_fill_VSX() test.\n\n");


    free(x_standard);
    free(x_optimized);
}


void test_THFloatVector_fill_VSX()
{
    clock_t start, end;
    double elapsedSeconds_optimized, elapsedSeconds_standard;

    float *x_standard  = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *x_optimized = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));

    float yVal0 = 17.2;
    float yVal1 = 8.2;
    float yVal2 = 5.1;
    float yVal3 = -0.9;

    //-------------------------------------------------
    // Performance Test
    //-------------------------------------------------
    start = clock();
    standardFloat_fill(x_standard, yVal0, VSX_PERF_NUM_TEST_ELEMENTS  );
    standardFloat_fill(x_standard, yVal1, VSX_PERF_NUM_TEST_ELEMENTS-1);
    standardFloat_fill(x_standard, yVal2, VSX_PERF_NUM_TEST_ELEMENTS-2);
    standardFloat_fill(x_standard, yVal3, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_standard = (double)(end - start) / CLOCKS_PER_SEC;
    printf("standardFloat_fill() test took %.5lf seconds\n", elapsedSeconds_standard);

    start = clock();
    THFloatVector_fill_VSX(x_optimized, yVal0, VSX_PERF_NUM_TEST_ELEMENTS  );
    THFloatVector_fill_VSX(x_optimized, yVal1, VSX_PERF_NUM_TEST_ELEMENTS-1);
    THFloatVector_fill_VSX(x_optimized, yVal2, VSX_PERF_NUM_TEST_ELEMENTS-2);
    THFloatVector_fill_VSX(x_optimized, yVal3, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_optimized = (double)(end - start) / CLOCKS_PER_SEC;
    printf("THFloatVector_fill_VSX() test took %.5lf seconds\n", elapsedSeconds_optimized);


    //-------------------------------------------------
    // Correctness Test
    //-------------------------------------------------
    yVal0 += 1.0;
    yVal1 += 1.0;
    yVal2 += 1.0;
    yVal3 -= 1.0;

    standardFloat_fill(    x_standard,  yVal0, VSX_FUNC_NUM_TEST_ELEMENTS);
    THFloatVector_fill_VSX(x_optimized, yVal0, VSX_FUNC_NUM_TEST_ELEMENTS);
    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
        assert(x_optimized[i] == yVal0);

    standardFloat_fill(    x_standard+1,  yVal1, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    THFloatVector_fill_VSX(x_optimized+1, yVal1, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    standardFloat_fill(    x_standard+2,  yVal2, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    THFloatVector_fill_VSX(x_optimized+2, yVal2, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    standardFloat_fill(    x_standard+3,  yVal3, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    THFloatVector_fill_VSX(x_optimized+3, yVal3, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    standardFloat_fill(    x_standard+517,  yVal0, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    THFloatVector_fill_VSX(x_optimized+517, yVal0, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    int r = rand() % 258;
    standardFloat_fill(    x_standard+517+r,  yVal2, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    THFloatVector_fill_VSX(x_optimized+517+r, yVal2, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
        assert(x_optimized[i] == x_standard[i]);
    printf("All assertions PASSED for THFloatVector_fill_VSX() test.\n\n");


    free(x_standard);
    free(x_optimized);
}


void test_THDoubleVector_cadd_VSX()
{
    clock_t start, end;
    double elapsedSeconds_optimized, elapsedSeconds_standard;

    double *z_standard  = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *z_optimized = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *x           = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *y           = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double c            = randDouble();

    // Initialize randomly
    for(int i = 0; i < VSX_PERF_NUM_TEST_ELEMENTS; i++)
    {
        x[i] = randDouble();
        y[i] = randDouble();
    }


    //-------------------------------------------------
    // Performance Test
    //-------------------------------------------------
    start = clock();
    standardDouble_cadd(z_standard, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    standardDouble_cadd(z_standard, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    standardDouble_cadd(z_standard, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    standardDouble_cadd(z_standard, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_standard = (double)(end - start) / CLOCKS_PER_SEC;
    printf("standardDouble_cadd() test took %.5lf seconds\n", elapsedSeconds_standard);

    start = clock();
    THDoubleVector_cadd_VSX(z_optimized, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    THDoubleVector_cadd_VSX(z_optimized, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    THDoubleVector_cadd_VSX(z_optimized, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    THDoubleVector_cadd_VSX(z_optimized, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_optimized = (double)(end - start) / CLOCKS_PER_SEC;
    printf("THDoubleVector_cadd_VSX() test took %.5lf seconds\n", elapsedSeconds_optimized);


    //-------------------------------------------------
    // Correctness Test
    //-------------------------------------------------
    standardDouble_cadd(    z_standard+1,  x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    THDoubleVector_cadd_VSX(z_optimized+1, x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    standardDouble_cadd(    z_standard+2,  x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    THDoubleVector_cadd_VSX(z_optimized+2, x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    standardDouble_cadd(    z_standard+3,  x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    THDoubleVector_cadd_VSX(z_optimized+3, x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    standardDouble_cadd(    z_standard+517,  x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    THDoubleVector_cadd_VSX(z_optimized+517, x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    int r = rand() % 258;
    standardDouble_cadd(    z_standard+517+r,  x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    THDoubleVector_cadd_VSX(z_optimized+517+r, x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
    {
        if(!near(z_optimized[i], z_standard[i]))
            printf("%d %f %f\n", i, z_optimized[i], z_standard[i]);
        assert(near(z_optimized[i], z_standard[i]));
    }
    printf("All assertions PASSED for THDoubleVector_cadd_VSX() test.\n\n");


    free(z_standard);
    free(z_optimized);
    free(x);
}

void test_THFloatVector_cadd_VSX()
{
    clock_t start, end;
    double elapsedSeconds_optimized, elapsedSeconds_standard;

    float *z_standard  = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *z_optimized = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *x           = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *y           = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float c            = (float)randDouble();

    // Initialize randomly
    for(int i = 0; i < VSX_PERF_NUM_TEST_ELEMENTS; i++)
    {
        x[i] = (float)randDouble();
        y[i] = (float)randDouble();
    }


    //-------------------------------------------------
    // Performance Test
    //-------------------------------------------------
    start = clock();
    standardFloat_cadd(z_standard, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    standardFloat_cadd(z_standard, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    standardFloat_cadd(z_standard, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    standardFloat_cadd(z_standard, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_standard = (double)(end - start) / CLOCKS_PER_SEC;
    printf("standardFloat_cadd() test took %.5lf seconds\n", elapsedSeconds_standard);

    start = clock();
    THFloatVector_cadd_VSX(z_optimized, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    THFloatVector_cadd_VSX(z_optimized, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    THFloatVector_cadd_VSX(z_optimized, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    THFloatVector_cadd_VSX(z_optimized, x, y, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_optimized = (double)(end - start) / CLOCKS_PER_SEC;
    printf("THFloatVector_cadd_VSX() test took %.5lf seconds\n", elapsedSeconds_optimized);


    //-------------------------------------------------
    // Correctness Test
    //-------------------------------------------------
    standardFloat_cadd(    z_standard+1,  x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    THFloatVector_cadd_VSX(z_optimized+1, x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    standardFloat_cadd(    z_standard+2,  x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    THFloatVector_cadd_VSX(z_optimized+2, x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    standardFloat_cadd(    z_standard+3,  x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    THFloatVector_cadd_VSX(z_optimized+3, x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    standardFloat_cadd(    z_standard+517,  x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    THFloatVector_cadd_VSX(z_optimized+517, x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    int r = rand() % 258;
    standardFloat_cadd(    z_standard+517+r,  x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    THFloatVector_cadd_VSX(z_optimized+517+r, x, y, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
    {
        if(!near(z_optimized[i], z_standard[i]))
            printf("%d %f %f\n", i, z_optimized[i], z_standard[i]);
        assert(near(z_optimized[i], z_standard[i]));
    }
    printf("All assertions PASSED for THFloatVector_cadd_VSX() test.\n\n");


    free(z_standard);
    free(z_optimized);
    free(x);
}

void test_THDoubleVector_adds_VSX()
{
    clock_t start, end;
    double elapsedSeconds_optimized, elapsedSeconds_standard;

    double *y_standard  = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *y_optimized = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *x           = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double c            = randDouble();

    // Initialize randomly
    for(int i = 0; i < VSX_PERF_NUM_TEST_ELEMENTS; i++)
        x[i] = randDouble();

    //-------------------------------------------------
    // Performance Test
    //-------------------------------------------------
    start = clock();
    standardDouble_adds(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    standardDouble_adds(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    standardDouble_adds(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    standardDouble_adds(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_standard = (double)(end - start) / CLOCKS_PER_SEC;
    printf("standardDouble_adds() test took %.5lf seconds\n", elapsedSeconds_standard);

    start = clock();
    THDoubleVector_adds_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    THDoubleVector_adds_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    THDoubleVector_adds_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    THDoubleVector_adds_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_optimized = (double)(end - start) / CLOCKS_PER_SEC;
    printf("THDoubleVector_adds_VSX() test took %.5lf seconds\n", elapsedSeconds_optimized);


    //-------------------------------------------------
    // Correctness Test
    //-------------------------------------------------
    standardDouble_adds(    y_standard+1,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    THDoubleVector_adds_VSX(y_optimized+1, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    standardDouble_adds(    y_standard+2,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    THDoubleVector_adds_VSX(y_optimized+2, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    standardDouble_adds(    y_standard+3,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    THDoubleVector_adds_VSX(y_optimized+3, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    standardDouble_adds(    y_standard+517,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    THDoubleVector_adds_VSX(y_optimized+517, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    int r = rand() % 258;
    standardDouble_adds(    y_standard+517+r,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    THDoubleVector_adds_VSX(y_optimized+517+r, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
    {
        if(!near(y_optimized[i], y_standard[i]))
            printf("%d %f %f\n", i, y_optimized[i], y_standard[i]);
        assert(near(y_optimized[i], y_standard[i]));
    }
    printf("All assertions PASSED for THDoubleVector_adds_VSX() test.\n\n");


    free(y_standard);
    free(y_optimized);
    free(x);
}


void test_THFloatVector_adds_VSX()
{
    clock_t start, end;
    double elapsedSeconds_optimized, elapsedSeconds_standard;

    float *y_standard  = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *y_optimized = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *x           = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float c            = (float)randDouble();

    // Initialize randomly
    for(int i = 0; i < VSX_PERF_NUM_TEST_ELEMENTS; i++)
        x[i] = (float)randDouble();


    //-------------------------------------------------
    // Performance Test
    //-------------------------------------------------
    start = clock();
    standardFloat_adds(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    standardFloat_adds(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    standardFloat_adds(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    standardFloat_adds(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_standard = (double)(end - start) / CLOCKS_PER_SEC;
    printf("standardFloat_adds() test took %.5lf seconds\n", elapsedSeconds_standard);

    start = clock();
    THFloatVector_adds_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    THFloatVector_adds_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    THFloatVector_adds_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    THFloatVector_adds_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_optimized = (double)(end - start) / CLOCKS_PER_SEC;
    printf("THFloatVector_adds_VSX() test took %.5lf seconds\n", elapsedSeconds_optimized);


    //-------------------------------------------------
    // Correctness Test
    //-------------------------------------------------
    standardFloat_adds(    y_standard+1,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    THFloatVector_adds_VSX(y_optimized+1, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    standardFloat_adds(    y_standard+2,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    THFloatVector_adds_VSX(y_optimized+2, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    standardFloat_adds(    y_standard+3,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    THFloatVector_adds_VSX(y_optimized+3, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    standardFloat_adds(    y_standard+517,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    THFloatVector_adds_VSX(y_optimized+517, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    int r = rand() % 258;
    standardFloat_adds(    y_standard+517+r,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    THFloatVector_adds_VSX(y_optimized+517+r, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
    {
        if(!near(y_optimized[i], y_standard[i]))
            printf("%d %f %f\n", i, y_optimized[i], y_standard[i]);
        assert(near(y_optimized[i], y_standard[i]));
    }
    printf("All assertions PASSED for THFloatVector_adds_VSX() test.\n\n");


    free(y_standard);
    free(y_optimized);
    free(x);
}


void test_THDoubleVector_cmul_VSX()
{
    clock_t start, end;
    double elapsedSeconds_optimized, elapsedSeconds_standard;

    double *z_standard  = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *z_optimized = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *x           = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *y           = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));

    // Initialize randomly
    for(int i = 0; i < VSX_PERF_NUM_TEST_ELEMENTS; i++)
    {
        x[i] = randDouble();
        y[i] = randDouble();
    }


    //-------------------------------------------------
    // Performance Test
    //-------------------------------------------------
    start = clock();
    standardDouble_cmul(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS  );
    standardDouble_cmul(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS-1);
    standardDouble_cmul(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS-2);
    standardDouble_cmul(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_standard = (double)(end - start) / CLOCKS_PER_SEC;
    printf("standardDouble_cmul() test took %.5lf seconds\n", elapsedSeconds_standard);

    start = clock();
    THDoubleVector_cmul_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS  );
    THDoubleVector_cmul_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS-1);
    THDoubleVector_cmul_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS-2);
    THDoubleVector_cmul_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_optimized = (double)(end - start) / CLOCKS_PER_SEC;
    printf("THDoubleVector_cmul_VSX() test took %.5lf seconds\n", elapsedSeconds_optimized);


    //-------------------------------------------------
    // Correctness Test
    //-------------------------------------------------
    standardDouble_cmul(    z_standard+1,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    THDoubleVector_cmul_VSX(z_optimized+1, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    standardDouble_cmul(    z_standard+2,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    THDoubleVector_cmul_VSX(z_optimized+2, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    standardDouble_cmul(    z_standard+3,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    THDoubleVector_cmul_VSX(z_optimized+3, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    standardDouble_cmul(    z_standard+517,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    THDoubleVector_cmul_VSX(z_optimized+517, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    int r = rand() % 258;
    standardDouble_cmul(    z_standard+517+r,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    THDoubleVector_cmul_VSX(z_optimized+517+r, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
    {
        if(!near(z_optimized[i], z_standard[i]))
            printf("%d %f %f\n", i, z_optimized[i], z_standard[i]);
        assert(near(z_optimized[i], z_standard[i]));
    }
    printf("All assertions PASSED for THDoubleVector_cmul_VSX() test.\n\n");


    free(z_standard);
    free(z_optimized);
    free(x);
}

void test_THFloatVector_cmul_VSX()
{
    clock_t start, end;
    double elapsedSeconds_optimized, elapsedSeconds_standard;

    float *z_standard  = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *z_optimized = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *x           = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *y           = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));

    // Initialize randomly
    for(int i = 0; i < VSX_PERF_NUM_TEST_ELEMENTS; i++)
    {
        x[i] = (float)randDouble();
        y[i] = (float)randDouble();
    }


    //-------------------------------------------------
    // Performance Test
    //-------------------------------------------------
    start = clock();
    standardFloat_cmul(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS  );
    standardFloat_cmul(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS-1);
    standardFloat_cmul(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS-2);
    standardFloat_cmul(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_standard = (double)(end - start) / CLOCKS_PER_SEC;
    printf("standardFloat_cmul() test took %.5lf seconds\n", elapsedSeconds_standard);

    start = clock();
    THFloatVector_cmul_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS  );
    THFloatVector_cmul_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS-1);
    THFloatVector_cmul_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS-2);
    THFloatVector_cmul_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_optimized = (double)(end - start) / CLOCKS_PER_SEC;
    printf("THFloatVector_cmul_VSX() test took %.5lf seconds\n", elapsedSeconds_optimized);


    //-------------------------------------------------
    // Correctness Test
    //-------------------------------------------------
    standardFloat_cmul(    z_standard+1,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    THFloatVector_cmul_VSX(z_optimized+1, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    standardFloat_cmul(    z_standard+2,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    THFloatVector_cmul_VSX(z_optimized+2, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    standardFloat_cmul(    z_standard+3,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    THFloatVector_cmul_VSX(z_optimized+3, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    standardFloat_cmul(    z_standard+517,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    THFloatVector_cmul_VSX(z_optimized+517, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    int r = rand() % 258;
    standardFloat_cmul(    z_standard+517+r,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    THFloatVector_cmul_VSX(z_optimized+517+r, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
    {
        if(!near(z_optimized[i], z_standard[i]))
            printf("%d %f %f\n", i, z_optimized[i], z_standard[i]);
        assert(near(z_optimized[i], z_standard[i]));
    }
    printf("All assertions PASSED for THFloatVector_cmul_VSX() test.\n\n");


    free(z_standard);
    free(z_optimized);
    free(x);
}

void test_THDoubleVector_muls_VSX()
{
    clock_t start, end;
    double elapsedSeconds_optimized, elapsedSeconds_standard;

    double *y_standard  = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *y_optimized = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *x           = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double c            = randDouble();

    // Initialize randomly
    for(int i = 0; i < VSX_PERF_NUM_TEST_ELEMENTS; i++)
    {
        x[i] = randDouble();
    }


    //-------------------------------------------------
    // Performance Test
    //-------------------------------------------------
    start = clock();
    standardDouble_muls(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    standardDouble_muls(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    standardDouble_muls(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    standardDouble_muls(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_standard = (double)(end - start) / CLOCKS_PER_SEC;
    printf("standardDouble_muls() test took %.5lf seconds\n", elapsedSeconds_standard);

    start = clock();
    THDoubleVector_muls_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    THDoubleVector_muls_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    THDoubleVector_muls_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    THDoubleVector_muls_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_optimized = (double)(end - start) / CLOCKS_PER_SEC;
    printf("THDoubleVector_muls_VSX() test took %.5lf seconds\n", elapsedSeconds_optimized);


    //-------------------------------------------------
    // Correctness Test
    //-------------------------------------------------
    standardDouble_muls(    y_standard+1,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    THDoubleVector_muls_VSX(y_optimized+1, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    standardDouble_muls(    y_standard+2,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    THDoubleVector_muls_VSX(y_optimized+2, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    standardDouble_muls(    y_standard+3,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    THDoubleVector_muls_VSX(y_optimized+3, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    standardDouble_muls(    y_standard+517,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    THDoubleVector_muls_VSX(y_optimized+517, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    int r = rand() % 258;
    standardDouble_muls(    y_standard+517+r,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    THDoubleVector_muls_VSX(y_optimized+517+r, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));

    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
    {
        if(!near(y_optimized[i], y_standard[i]))
            printf("%d %f %f\n", i, y_optimized[i], y_standard[i]);
        assert(near(y_optimized[i], y_standard[i]));
    }
    printf("All assertions PASSED for THDoubleVector_muls_VSX() test.\n\n");


    free(y_standard);
    free(y_optimized);
    free(x);
}

void test_THFloatVector_muls_VSX()
{
    clock_t start, end;
    double elapsedSeconds_optimized, elapsedSeconds_standard;

    float *y_standard  = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *y_optimized = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *x           = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float c           = (float)randDouble();

    // Initialize randomly
    for(int i = 0; i < VSX_PERF_NUM_TEST_ELEMENTS; i++)
    {
        x[i] = (float)randDouble();
    }


    //-------------------------------------------------
    // Performance Test
    //-------------------------------------------------
    start = clock();
    standardFloat_muls(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    standardFloat_muls(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    standardFloat_muls(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    standardFloat_muls(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_standard = (double)(end - start) / CLOCKS_PER_SEC;
    printf("standardFloat_muls() test took %.5lf seconds\n", elapsedSeconds_standard);

    start = clock();
    THFloatVector_muls_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    THFloatVector_muls_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    THFloatVector_muls_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    THFloatVector_muls_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_optimized = (double)(end - start) / CLOCKS_PER_SEC;
    printf("THFloatVector_muls_VSX() test took %.5lf seconds\n", elapsedSeconds_optimized);


    //-------------------------------------------------
    // Correctness Test
    //-------------------------------------------------
    standardFloat_muls(    y_standard+1,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    THFloatVector_muls_VSX(y_optimized+1, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    standardFloat_muls(    y_standard+2,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    THFloatVector_muls_VSX(y_optimized+2, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    standardFloat_muls(    y_standard+3,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    THFloatVector_muls_VSX(y_optimized+3, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    standardFloat_muls(    y_standard+517,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    THFloatVector_muls_VSX(y_optimized+517, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    int r = rand() % 258;
    standardFloat_muls(    y_standard+517+r,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    THFloatVector_muls_VSX(y_optimized+517+r, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
    {
        if(!near(y_optimized[i], y_standard[i]))
            printf("%d %f %f\n", i, y_optimized[i], y_standard[i]);
        assert(near(y_optimized[i], y_standard[i]));
    }
    printf("All assertions PASSED for THFloatVector_muls_VSX() test.\n\n");


    free(y_standard);
    free(y_optimized);
    free(x);
}



void test_THDoubleVector_cdiv_VSX()
{
    clock_t start, end;
    double elapsedSeconds_optimized, elapsedSeconds_standard;

    double *z_standard  = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *z_optimized = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *x           = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *y           = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));

    // Initialize randomly
    for(int i = 0; i < VSX_PERF_NUM_TEST_ELEMENTS; i++)
    {
        x[i] = randDouble();
        y[i] = randDouble();
    }


    //-------------------------------------------------
    // Performance Test
    //-------------------------------------------------
    start = clock();
    standardDouble_cdiv(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS  );
    standardDouble_cdiv(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS-1);
    standardDouble_cdiv(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS-2);
    standardDouble_cdiv(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_standard = (double)(end - start) / CLOCKS_PER_SEC;
    printf("standardDouble_cdiv() test took %.5lf seconds\n", elapsedSeconds_standard);

    start = clock();
    THDoubleVector_cdiv_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS  );
    THDoubleVector_cdiv_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS-1);
    THDoubleVector_cdiv_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS-2);
    THDoubleVector_cdiv_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_optimized = (double)(end - start) / CLOCKS_PER_SEC;
    printf("THDoubleVector_cdiv_VSX() test took %.5lf seconds\n", elapsedSeconds_optimized);


    //-------------------------------------------------
    // Correctness Test
    //-------------------------------------------------
    standardDouble_cdiv(    z_standard+1,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    THDoubleVector_cdiv_VSX(z_optimized+1, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    standardDouble_cdiv(    z_standard+2,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    THDoubleVector_cdiv_VSX(z_optimized+2, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    standardDouble_cdiv(    z_standard+3,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    THDoubleVector_cdiv_VSX(z_optimized+3, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    standardDouble_cdiv(    z_standard+517,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    THDoubleVector_cdiv_VSX(z_optimized+517, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    int r = rand() % 258;
    standardDouble_cdiv(    z_standard+517+r,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    THDoubleVector_cdiv_VSX(z_optimized+517+r, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
    {
        if(!near(z_optimized[i], z_standard[i]))
            printf("%d %f %f\n", i, z_optimized[i], z_standard[i]);
        assert(near(z_optimized[i], z_standard[i]));
    }
    printf("All assertions PASSED for THDoubleVector_cdiv_VSX() test.\n\n");


    free(z_standard);
    free(z_optimized);
    free(x);
}

void test_THFloatVector_cdiv_VSX()
{
    clock_t start, end;
    double elapsedSeconds_optimized, elapsedSeconds_standard;

    float *z_standard  = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *z_optimized = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *x           = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *y           = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));

    // Initialize randomly
    for(int i = 0; i < VSX_PERF_NUM_TEST_ELEMENTS; i++)
    {
        x[i] = (float)randDouble();
        y[i] = (float)randDouble();
    }


    //-------------------------------------------------
    // Performance Test
    //-------------------------------------------------
    start = clock();
    standardFloat_cdiv(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS  );
    standardFloat_cdiv(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS-1);
    standardFloat_cdiv(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS-2);
    standardFloat_cdiv(z_standard, x, y, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_standard = (double)(end - start) / CLOCKS_PER_SEC;
    printf("standardFloat_cdiv() test took %.5lf seconds\n", elapsedSeconds_standard);

    start = clock();
    THFloatVector_cdiv_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS  );
    THFloatVector_cdiv_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS-1);
    THFloatVector_cdiv_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS-2);
    THFloatVector_cdiv_VSX(z_optimized, x, y, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_optimized = (double)(end - start) / CLOCKS_PER_SEC;
    printf("THFloatVector_cdiv_VSX() test took %.5lf seconds\n", elapsedSeconds_optimized);


    //-------------------------------------------------
    // Correctness Test
    //-------------------------------------------------
    standardFloat_cdiv(    z_standard+1,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    THFloatVector_cdiv_VSX(z_optimized+1, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    standardFloat_cdiv(    z_standard+2,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    THFloatVector_cdiv_VSX(z_optimized+2, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    standardFloat_cdiv(    z_standard+3,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    THFloatVector_cdiv_VSX(z_optimized+3, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    standardFloat_cdiv(    z_standard+517,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    THFloatVector_cdiv_VSX(z_optimized+517, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    int r = rand() % 258;
    standardFloat_cdiv(    z_standard+517+r,  x, y, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    THFloatVector_cdiv_VSX(z_optimized+517+r, x, y, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
    {
        if(!near(z_optimized[i], z_standard[i]))
            printf("%d %f %f\n", i, z_optimized[i], z_standard[i]);
        assert(near(z_optimized[i], z_standard[i]));
    }
    printf("All assertions PASSED for THFloatVector_cdiv_VSX() test.\n\n");


    free(z_standard);
    free(z_optimized);
    free(x);
}

void test_THDoubleVector_divs_VSX()
{
    clock_t start, end;
    double elapsedSeconds_optimized, elapsedSeconds_standard;

    double *y_standard  = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *y_optimized = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double *x           = (double *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(double));
    double c            = randDouble();

    // Initialize randomly
    for(int i = 0; i < VSX_PERF_NUM_TEST_ELEMENTS; i++)
    {
        x[i] = randDouble();
    }


    //-------------------------------------------------
    // Performance Test
    //-------------------------------------------------
    start = clock();
    standardDouble_divs(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    standardDouble_divs(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    standardDouble_divs(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    standardDouble_divs(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_standard = (double)(end - start) / CLOCKS_PER_SEC;
    printf("standardDouble_divs() test took %.5lf seconds\n", elapsedSeconds_standard);

    start = clock();
    THDoubleVector_divs_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    THDoubleVector_divs_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    THDoubleVector_divs_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    THDoubleVector_divs_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_optimized = (double)(end - start) / CLOCKS_PER_SEC;
    printf("THDoubleVector_divs_VSX() test took %.5lf seconds\n", elapsedSeconds_optimized);


    //-------------------------------------------------
    // Correctness Test
    //-------------------------------------------------
    standardDouble_divs(    y_standard+1,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    THDoubleVector_divs_VSX(y_optimized+1, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    standardDouble_divs(    y_standard+2,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    THDoubleVector_divs_VSX(y_optimized+2, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    standardDouble_divs(    y_standard+3,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    THDoubleVector_divs_VSX(y_optimized+3, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    standardDouble_divs(    y_standard+517,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    THDoubleVector_divs_VSX(y_optimized+517, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    int r = rand() % 258;
    standardDouble_divs(    y_standard+517+r,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    THDoubleVector_divs_VSX(y_optimized+517+r, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));

    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
    {
        if(!near(y_optimized[i], y_standard[i]))
            printf("%d %f %f\n", i, y_optimized[i], y_standard[i]);
        assert(near(y_optimized[i], y_standard[i]));
    }
    printf("All assertions PASSED for THDoubleVector_divs_VSX() test.\n\n");


    free(y_standard);
    free(y_optimized);
    free(x);
}

void test_THFloatVector_divs_VSX()
{
    clock_t start, end;
    double elapsedSeconds_optimized, elapsedSeconds_standard;

    float *y_standard  = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *y_optimized = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float *x           = (float *)malloc(VSX_PERF_NUM_TEST_ELEMENTS*sizeof(float));
    float c            = (float)randDouble();

    // Initialize randomly
    for(int i = 0; i < VSX_PERF_NUM_TEST_ELEMENTS; i++)
    {
        x[i] = (float)randDouble();
    }


    //-------------------------------------------------
    // Performance Test
    //-------------------------------------------------
    start = clock();
    standardFloat_divs(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    standardFloat_divs(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    standardFloat_divs(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    standardFloat_divs(y_standard, x, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_standard = (double)(end - start) / CLOCKS_PER_SEC;
    printf("standardFloat_divs() test took %.5lf seconds\n", elapsedSeconds_standard);

    start = clock();
    THFloatVector_divs_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS  );
    THFloatVector_divs_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-1);
    THFloatVector_divs_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-2);
    THFloatVector_divs_VSX(y_optimized, x, c, VSX_PERF_NUM_TEST_ELEMENTS-3);
    end = clock();

    elapsedSeconds_optimized = (double)(end - start) / CLOCKS_PER_SEC;
    printf("THFloatVector_divs_VSX() test took %.5lf seconds\n", elapsedSeconds_optimized);


    //-------------------------------------------------
    // Correctness Test
    //-------------------------------------------------
    standardFloat_divs(    y_standard+1,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    THFloatVector_divs_VSX(y_optimized+1, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-2);
    standardFloat_divs(    y_standard+2,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    THFloatVector_divs_VSX(y_optimized+2, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-4);
    standardFloat_divs(    y_standard+3,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    THFloatVector_divs_VSX(y_optimized+3, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-6);
    standardFloat_divs(    y_standard+517,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    THFloatVector_divs_VSX(y_optimized+517, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-1029);
    int r = rand() % 258;
    standardFloat_divs(    y_standard+517+r,  x, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));
    THFloatVector_divs_VSX(y_optimized+517+r, x, c, VSX_FUNC_NUM_TEST_ELEMENTS-(1029+r+100));

    for(int i = 0; i < VSX_FUNC_NUM_TEST_ELEMENTS; i++)
    {
        if(!near(y_optimized[i], y_standard[i]))
            printf("%d %f %f\n", i, y_optimized[i], y_standard[i]);
        assert(near(y_optimized[i], y_standard[i]));
    }
    printf("All assertions PASSED for THFloatVector_divs_VSX() test.\n\n");


    free(y_standard);
    free(y_optimized);
    free(x);
}


//--------------------------------------------------------------------------------------------------
// Run tests:
//--------------------------------------------------------------------------------------------------
int main()
{
    printf("\n");


    // First test utility functions

    assert(!near(0.1, -0.1));
    assert(!near(0.1f, -0.1f));
    assert(!near(9, 10));
    assert(near(0.1, 0.1000001));
    assert(near(0.1f, 0.1000001f));
    assert(near(100.764, 100.764));
    assert(!near(NAN, 0.0));
    assert(!near(-9.5, NAN));
    assert(!near(NAN, 100));
    assert(!near(-0.0, NAN));
    assert(near(NAN, NAN));
    assert(near(INFINITY, INFINITY));
    assert(near(-INFINITY, -INFINITY));
    assert(!near(INFINITY, NAN));
    assert(!near(0, INFINITY));
    assert(!near(-999.4324, INFINITY));
    assert(!near(INFINITY, 982374.1));
    assert(!near(-INFINITY, INFINITY));



    // Then test each vectorized function

    test_THDoubleVector_fill_VSX();
    test_THFloatVector_fill_VSX();

    test_THDoubleVector_cadd_VSX();
    test_THFloatVector_cadd_VSX();

    test_THDoubleVector_adds_VSX();
    test_THFloatVector_adds_VSX();

    test_THDoubleVector_cmul_VSX();
    test_THFloatVector_cmul_VSX();

    test_THDoubleVector_muls_VSX();
    test_THFloatVector_muls_VSX();

    test_THDoubleVector_cdiv_VSX();
    test_THFloatVector_cdiv_VSX();

    test_THDoubleVector_divs_VSX();
    test_THFloatVector_divs_VSX();



    printf("Finished runnning all tests. All tests PASSED.\n");
    return 0;
}


#endif  // defined RUN_VSX_TESTS

#endif  // defined __PPC64__

