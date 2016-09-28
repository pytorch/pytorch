#include <TH/TH.h>

void bad_func(THFloatTensor *tensor, int a, float b)
{
  THFloatTensor_mul(tensor, tensor, a);
  THFloatTensor_add(tensor, tensor, b);
  THFloatTensor_addbmm(tensor, 1, tensor, 1, tensor, tensor);
}
