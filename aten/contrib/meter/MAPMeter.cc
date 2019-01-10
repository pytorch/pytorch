#include "MAPMeter.h"

using namespace at;

MAPMeter::MAPMeter() {
   reset();
}

void MAPMeter::reset() {
   meter_.reset();
}

void MAPMeter::add(Tensor& output, Tensor& target) {
   meter_.add(output, target);
}

void MAPMeter::value(Tensor& val) {
   //TODO: 0-dim
   val.resize_({1});
   Tensor allvalues = val.type().tensor();
   meter_.value(allvalues);
   val.fill_(mean(allvalues));
}
