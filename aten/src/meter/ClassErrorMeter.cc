#include "ClassErrorMeter.h"
#include "ATen/ATen.h"
#include <cassert>

using namespace at;

ClassErrorMeter::ClassErrorMeter() {
   ClassErrorMeter(1);
}

ClassErrorMeter::ClassErrorMeter(const int64_t topk) {
   topkval_ = CPU(kShort).tensor();
   sumval_ = CPU(kShort).tensor();
   topkval_.resize_({topk});
   sumval_.resize_({topk});
   reset();
}

void ClassErrorMeter::reset() {
  range_out(topkval_, 1, numel(topkval_));
  sumval_.fill_(0.);
  n_ = 0;
}

void ClassErrorMeter::add(Tensor& output, Tensor& target) {

   // assertions and allocations:
   assert(output.dim() == 2 && target.dim() == 1);
   //assert(isSameSizeAs(output, target));
   auto sumval_d = sumval_.data<int16_t>();
   auto target_long = target.contiguous().toType(CPU(kLong));
   auto target_d = target_long.data<int64_t>();
   // update counts:
   Tensor val, idx;
   std::tie(val, idx) = topk(output, numel(topkval_), 1, true, true);
   for(uint64_t n = 0; n < output.size(0); ++n) {
      bool targetseen = false;
      Tensor idx_n = idx.select(0,n);
      auto idx_n_d = idx_n.data<int64_t>();
      for(uint64_t k = 0; k < numel(topkval_); ++k) {
         n_++;
         if(targetseen) {
            sumval_d[k]++;
         } else if(idx_n_d[k] == target_d[n]) {
            targetseen = true;
            sumval_d[k]++;
         }
      }
   }
}

void ClassErrorMeter::value(Tensor& val) {
   val.resize_({numel(topkval_)});
   auto val_d = val.data<double>();
   auto sumval_d = sumval_.data<int16_t>();
   for(uint64_t k = 0; k < numel(topkval_); ++k) {
     val_d[k] = 1.0 - (double(sumval_d[k]) / double(n_));
   }
}
