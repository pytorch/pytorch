#ifndef XT_AUC_METER_H
#define XT_AUC_METER_H

#include "Meter.h"
#include "APMeter.h"
#include "TensorLib/TensorLib.h"

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
