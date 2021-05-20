#include "observers/observer_config.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
int ObserverConfig::netInitSampleRate_ = 0;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
int ObserverConfig::netFollowupSampleRate_ = 0;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
int ObserverConfig::netFollowupSampleCount_ = 0;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
int ObserverConfig::operatorNetSampleRatio_ = 0;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
int ObserverConfig::skipIters_ = 0;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
unique_ptr<NetObserverReporter> ObserverConfig::reporter_ = nullptr;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
int ObserverConfig::marker_ = -1;
}
