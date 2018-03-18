#include "AUCMeter.h"
#include "APMeter.h"
#include <cassert>

using namespace at;

AUCMeter::AUCMeter() {
   reset();
}

void AUCMeter::reset() {
   meter_ = APMeter();
}

void AUCMeter::add(Tensor& output, Tensor& target) {
   meter_.add(output, target);
}

void AUCMeter::value(Tensor& val) {

   // get data from APMeter:
   Tensor outputs = meter_.getOutputs();
   Tensor targets = meter_.getTargets();

   // sort scores:
   Tensor sortval, sortidx, sorttgt;
   std::tie(sortval, sortidx) = sort(outputs, 0, true);
   sorttgt = index_select(targets, 0, sortidx);
   int64_t * sortidx_d = sortidx.data<int64_t>();
   int16_t * targets_d = sortidx.data<int16_t>();
   // construct the ROC curve:
   Tensor tpr = zeros(CPU(kDouble), {numel(outputs)});
   Tensor fpr = zeros(CPU(kDouble), {numel(outputs)});

   double * tpr_d = tpr.data<double>();
   double * fpr_d = fpr.data<double>();
   for(uint64_t n = 1; n <= numel(outputs); ++n) {
      if(targets_d[sortidx_d[n - 1]] == 1) {
         tpr_d[n] = tpr_d[n - 1] + 1.;
         fpr_d[n] = fpr_d[n - 1];
      } else {
         tpr_d[n] = tpr_d[n - 1];
         fpr_d[n] = fpr_d[n - 1] + 1.;
      }
   }
   tpr.div_(sum(targets));
   fpr.div_(sum(at::add(mul(targets, -1.), 1.)));

   /**
   local auc = torch.cmul(
      tpr:narrow(1, 1, tpr:nElement() - 1),
      fpr:narrow(1, 2, fpr:nElement() - 1) -
      fpr:narrow(1, 1, fpr:nElement() - 1)):sum()
   */

   val.resize_({1}).fill_(
     sum(mul(tpr.narrow(0, 0, numel(tpr) - 1),
                sub(fpr.narrow(0, 1, numel(tpr) - 1),
                     fpr.narrow(0, 0, numel(tpr) - 1)))));
}
