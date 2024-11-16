#pragma once

const char* UNARY_KERNEL_TEMPLATE = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float a[4] = {{0.886226899, -1.645349621, 0.914624893, -0.140543331}};
constant float b[4] = {{-2.118377725, 1.442710462, -0.329097515, 0.012229801}};
constant float c[4] = {{-1.970840454, -1.624906493, 3.429567803, 1.641345311}};
constant float d[2] = {{3.543889200, 1.637067800}};

kernel void erfinv_kernel( device {0} *output [[buffer(0)]],
                           device {1} *input [[buffer(1)]],
                           uint index [[thread_position_in_grid]]) {{

  float y = input[index];
  float x, z, num, dem; /*working variables */
  /* coefficients in rational expansion */

  float y_abs = abs(y);
  if (y_abs >= 1.0f) {{
    output[index] = {0}( y_abs > 1.0f ? NAN : copysign(INFINITY, y));
    return;
  }}
  if (y_abs <= 0.7f) {{
    z = y * y;
    num = ((a[3] * z + a[2]) * z + a[1])*z + a[0];
    dem = (((b[3] * z + b[2]) * z + b[1]) * z +b[0]) * z + 1.0f;
    x = y * num / dem;
  }} else {{
    z = sqrt(-1.0f*log((1.0-y_abs)/2.0));
    num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0];
    dem = (d[1] * z + d[0]) * z + 1.0f;
    x = copysign(num, y) / dem;
  }}

  output[index] = {0}(x);
}}

kernel void exp_kernel( device {0} *output [[buffer(0)]],
                        device {1} *input [[ buffer(1)]],
                        uint index [[thread_position_in_grid]]) {{
  output[index] = {0}(precise::exp(input[index]));
}}

kernel void exp_complex_kernel( device {0}2 *output [[buffer(0)]],
                                device {0}2 *input [[ buffer(1)]],
                                uint index [[thread_position_in_grid]]) {{
  output[index].x = {0}(precise::exp(input[index].x)*precise::cos(input[index].y));
  output[index].y = {0}(precise::exp(input[index].x)*precise::sin(input[index].y));
}}

kernel void tanh_kernel( device {0} *output [[buffer(0)]],
                        device {1} *input [[ buffer(1)]],
                        uint index [[thread_position_in_grid]]) {{
  output[index] = {0}(precise::tanh(input[index]));
}}


#if __METAL_VERSION__ >= 310
bfloat dot(bfloat2 a, bfloat2 b) {{
  return a.x * b.x + a.y * b.y;
}}
#endif

template<typename T>
T complex_div(T a, T b) {{
  auto denom = dot(b, b);
  return T(dot(a, b), a.y * b.x - a.x * b.y)/denom;
}}

kernel void tanh_complex_kernel( device {0}2 *output [[buffer(0)]],
                                 device {0}2 *input [[ buffer(1)]],
                                 uint index [[thread_position_in_grid]]) {{
  //tanh(x+iy)=(tanh(x)+itan(y))/(1+itahnh(x)*tan(y));
  auto tanh_x = {0}(precise::tanh(input[index].x));
  auto tan_y = {0}(precise::tan(input[index].y));
  output[index] = complex_div({0}2(tanh_x, tan_y), {0}2({0}(1), tanh_x * tan_y));
}}
)METAL";
