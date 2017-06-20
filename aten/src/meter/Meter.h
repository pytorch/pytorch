#ifndef XT_METER_H
#define XT_METER_H

#include "ATen/ATen.h"

using namespace at;

class Meter
{
public:
   virtual void add(Tensor& output, Tensor& target) = 0;
   virtual void value(Tensor& val) = 0;
   virtual void reset() = 0;
  virtual ~Meter() {};
};

#endif
