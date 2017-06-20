#ifndef XT_CLASS_ERROR_METER_H
#define XT_CLASS_ERROR_METER_H

#include "Meter.h"
#include "TensorLib/TensorLib.h"

class ClassErrorMeter : public Meter
{
public:
   ClassErrorMeter();
   ClassErrorMeter(const int64_t topk);
   virtual void reset();
   virtual void add(Tensor& output, Tensor& target);
   virtual void value(Tensor& val);
private:
   Tensor topkval_;
   Tensor sumval_;
   uint64_t n_;
};

#endif
