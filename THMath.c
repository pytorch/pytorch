#include <math.h>

#include "THMath.h"

double sigmoid(double value) {
  return 1.0 / (1.0 + exp(-value));
}

