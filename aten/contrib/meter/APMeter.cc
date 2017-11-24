#include "APMeter.h"
#include <math.h>
#include <cassert>

using namespace at;

APMeter::APMeter() {
   reset();
}

void APMeter::reset() {
   outputs_ = CPU(kFloat).tensor();
   targets_ = CPU(kFloat).tensor();
   n_ = 0;
}

void APMeter::add(Tensor& output, Tensor& target) {

   // assertions and allocations:
   assert(output.dim() == 2 && target.dim() == 2);
   //assert(isSameSizeAs(output, target));
   assert(output.size(1) == outputs_.size(1));

   // get current outputs and targets:
   Tensor curoutputs = getOutputs();
   Tensor curtargets = getTargets();

   // make sure underlying storages are sufficiently large:
   if(numel(outputs_) < numel(curoutputs) + numel(output)) {
      long long newsize = ceil(numel(outputs_) * 1.5);
      outputs_.resize_({newsize + numel(output)});
      targets_.resize_({newsize + numel(output)});
   }
   n_ += output.size(0);

   // store scores and targets:
   uint64_t offset = (numel(curoutputs) > 0) ? curoutputs.size(0) : 0;
   Tensor outputbuffer = outputs_.narrow( 0, offset, output.size(0));
   Tensor targetbuffer = targets_.narrow( 0, offset, target.size(0));

   outputbuffer.copy_(output);
   targetbuffer.copy_(target);
}

Tensor APMeter::getOutputs() {
   return outputs_.narrow(0, 0, n_);
}
Tensor APMeter::getTargets() {
   return targets_.narrow(0, 0, n_);
}

void APMeter::value(Tensor& val) {

   // get current outputs and targets:
   Tensor curoutputs = getOutputs();
   Tensor curtargets = getTargets();

   // allocate some memory:
   val.resize_({curoutputs.size(1)});
   double * val_d = val.data<double>();
   Tensor outputbuffer, targetbuffer, sortval, sortidx, sorttgt;
   Tensor truepos, precision;
   Tensor range = val.type().range(0,curoutputs.size(0));

   // loop over all classes:
   for(uint64_t k = 0; k < curoutputs.size(1); ++k) {

      // sort scores:
      outputbuffer = curoutputs.narrow( 1, k, 1);
      targetbuffer = curtargets.narrow(1, k, 1).contiguous().toType(CPU(kDouble));
      double * targetbuffer_d = targetbuffer.data<double>();
      std::tie(sortval, sortidx) = sort(curoutputs, 0);
      sorttgt = index_select(targetbuffer, 0, sortidx);

      // compue true positive sums, and precision:
      truepos = cumsum(targetbuffer,0);  // NOTE: Cast to double first?
      precision = div(truepos, range);
      double * precision_d = precision.data<double>();
      // compute average precision:
      val_d[k] = .0;
      for(uint64_t n = 0; n < precision.size(0); ++n) {
         if(targetbuffer_d[n] != 0.)
            val_d[k] += precision_d[n];
      }
      auto norm = sum(targetbuffer).toCDouble();
      if(norm > 0)
        val_d[k] /= norm;
   }
}
