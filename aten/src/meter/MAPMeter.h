#ifndef XT_MAP_METER_H
#define XT_MAP_METER_H

#include "Meter.h"
#include "APMeter.h"
#include "TensorLib/TensorLib.h"

class MAPMeter : public Meter
{
public:
   MAPMeter();
   virtual void reset();
   virtual void add(Tensor& output, Tensor& target);
   virtual void value(Tensor& val);
private:
   APMeter meter_;
};

#endif
