#pragma once

template <typename T>
struct ELUParams {
  T alpha;
  T scale;
  T input_scale;
};

template <typename T>
struct ELUBackwardParams {
  T alpha;
  T scale;
  T input_scale;
  bool is_result;
};
