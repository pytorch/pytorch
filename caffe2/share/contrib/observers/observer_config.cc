#include "caffe2/share/contrib/observers/observer_config.h"

namespace caffe2 {

int ObserverConfig::netSampleRate_ = 1;
int ObserverConfig::operatorNetSampleRatio_ = 0;
int ObserverConfig::skipIters_ = 10;

unique_ptr<NetObserverReporter> ObserverConfig::reporter_ = nullptr;
}
