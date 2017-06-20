#include "APMeter.h"
#include <iostream>

using namespace tlib;

int main()
{
   auto && T = CPU(kFloat);
   std::cout << "hello\n";
   APMeter meter;
   Tensor output = T.randn({10, 7});
   Tensor target = T.zeros({10, 7});
   for(uint64_t n = 0; n < 10; ++n) {
     Tensor row = target.select(0,n);
     auto row_d = row.data<float>();
     row_d[rand() % 7] = 1.;
   }
   std::cout << output;
   std::cout << target;
   meter.add(output, target);
   Tensor val;
   meter.value(val);
   std::cout << "value: " << val << std::endl;
   return 0;
}
