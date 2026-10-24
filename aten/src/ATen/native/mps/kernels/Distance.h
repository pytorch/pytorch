#pragma once

struct CdistFwdParams {
  long B;
  long P;
  long R;
  long D;
  float p;
};

struct CdistBwdParams {
  long B;
  long P;
  long R;
  long D;
  float p_minus_1;
};
