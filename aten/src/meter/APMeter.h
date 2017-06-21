#ifndef AT_AP_METER_H
#define AT_AP_METER_H

#include "Meter.h"
#include "ATen/ATen.h"

class APMeter : public Meter
{
public:
   APMeter();
   virtual void add(Tensor& output, Tensor& target);
   virtual void value(Tensor& val);
   virtual void reset();
   virtual Tensor getOutputs();
   virtual Tensor getTargets();
private:
   Tensor outputs_;
   Tensor targets_;
   uint64_t n_;
};

#endif
