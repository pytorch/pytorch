#ifndef AT_CLASS_ERROR_METER_H
#define AT_CLASS_ERROR_METER_H

#include "Meter.h"
#include "ATen/ATen.h"

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
