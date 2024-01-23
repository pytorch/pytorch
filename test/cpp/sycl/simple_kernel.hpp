#pragma once

// Create an idx array on SYCL GPU device
// res      - host buffer for result
// numel    - length of the idx array
void itoa(float* res, int numel);
