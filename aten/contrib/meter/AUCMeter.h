#ifndef AT_AUC_METER_H
#define AT_AUC_METER_H

#include "Meter.h"
#include "APMeter.h"
#include "ATen/ATen.h"

class AUCMeter : public Meter
{
public:
   AUCMeter();
   virtual void reset();
   virtual void add(Tensor& output, Tensor& target);
   virtual void value(Tensor& val);
private:
   APMeter meter_;
};

#endif
