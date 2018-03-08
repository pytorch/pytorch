#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensorRandom.cpp"
#else

using namespace thd;
using namespace rpc;
using namespace master;

void THDTensor_(random)(THDTensor *self, THDGenerator *_generator) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorRandom, self, _generator),
    THDState::s_current_worker
  );
}

void THDTensor_(geometric)(THDTensor *self, THDGenerator *_generator, double p) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorGeometric, self, _generator, p),
    THDState::s_current_worker
  );
}

void THDTensor_(bernoulli)(THDTensor *self, THDGenerator *_generator, double p) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorBernoulli, self, _generator, p),
    THDState::s_current_worker
  );
}

void THDTensor_(bernoulli_FloatTensor)(THDTensor *self, THDGenerator *_generator,
                                       THDFloatTensor *p) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorBernoulli_FloatTensor, self, _generator, p),
    THDState::s_current_worker
  );
}

void THDTensor_(bernoulli_DoubleTensor)(THDTensor *self, THDGenerator *_generator,
                                        THDDoubleTensor *p) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorBernoulli_DoubleTensor, self, _generator, p),
    THDState::s_current_worker
  );
}

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

void THDTensor_(uniform)(THDTensor *self, THDGenerator *_generator, double a,
                         double b) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorUniform, self, _generator, a, b),
    THDState::s_current_worker
  );
}

void THDTensor_(normal)(THDTensor *self, THDGenerator *_generator, double mean,
                        double stdv) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorNormal, self, _generator, mean, stdv),
    THDState::s_current_worker
  );
}

void THDTensor_(exponential)(THDTensor *self, THDGenerator *_generator,
                             double lambda) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorExponential, self, _generator, lambda),
    THDState::s_current_worker
  );
}

void THDTensor_(cauchy)(THDTensor *self, THDGenerator *_generator, double median,
                        double sigma) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCauchy, self, _generator, median, sigma),
    THDState::s_current_worker
  );
}

void THDTensor_(logNormal)(THDTensor *self, THDGenerator *_generator, double mean,
                           double stdv) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorLogNormal, self, _generator, mean, stdv),
    THDState::s_current_worker
  );
}

void THDTensor_(truncatedNormal)(THDTensor *self, THDGenerator *_generator, double mean,
                           double stdv, double min_val, double max_val) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorTruncatedNormal, self, _generator, mean, stdv, min_val, max_val),
    THDState::s_current_worker
  );
}

void THDTensor_(multinomial)(THDLongTensor *self, THDGenerator *_generator,
                             THDTensor *prob_dist, int n_sample,
                             int with_replacement) {
  int start_dim = THDTensor_(nDimension)(prob_dist);
  if (start_dim == 1) {
    THDTensor_(resize2d)(prob_dist, 1, THDTensor_(size)(prob_dist, 0));
  }

  long n_dist = THDTensor_(size)(prob_dist, 0);
  long n_categories = THDTensor_(size)(prob_dist, 1);

  THArgCheck(n_sample > 0, 2, "cannot sample n_sample < 0 samples");

  if (!with_replacement) {
    THArgCheck((!with_replacement) && (n_sample <= n_categories), 2, \
    "cannot sample n_sample > prob_dist:size(1) samples without replacement");
  }

  /* will contain multinomial samples (category indices to be returned) */
  THDLongTensor_resize2d(self, n_dist, n_sample);

  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorMultinomial,
      self,
      _generator,
      prob_dist,
      n_sample,
      with_replacement
    ),
    THDState::s_current_worker
  );

  if (start_dim == 1) {
    THDLongTensor_resize1d(self, n_sample);
    THDTensor_(resize1d)(prob_dist, n_categories);
  }
}

#endif // defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

#endif // TH_GENERIC_FILE
