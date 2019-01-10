#ifndef AT_MSE_METER_H
#define AT_MSE_METER_H

#include "Meter.h"
#include "ATen/ATen.h"

class MSEMeter : public Meter
{
public:
   MSEMeter();
   virtual void reset();
   virtual void add(Tensor& output, Tensor& target);
   virtual void value(Tensor& val);
private:
   double val_;
   uint64_t n_;
};

#endif
