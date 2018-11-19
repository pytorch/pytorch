#include <TH/TH.h>

void good_func(THTensor *tensor, int a, float b)
{
  THFloatTensor_mul(tensor, tensor, a);
  THFloatTensor_add(tensor, tensor, b);
}

THTensor * new_tensor(int a)
{
  THTensor *t = THFloatTensor_newWithSize2d(a, a);
  THFloatTensor_fill(t, a);
  return t;
}

float int_to_float(int a)
{
  return a;
}
