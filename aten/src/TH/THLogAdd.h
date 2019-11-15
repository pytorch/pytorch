#ifndef TH_LOG_ADD_INC
#define TH_LOG_ADD_INC

#include <TH/THGeneral.h>

extern "C" TH_API const double THLog2Pi;
extern "C" TH_API const double THLogZero;
extern "C" TH_API const double THLogOne;

TH_API double THLogAdd(double log_a, double log_b);
TH_API double THLogSub(double log_a, double log_b);
TH_API double THExpMinusApprox(const double x);

#endif
