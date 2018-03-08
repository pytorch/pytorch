#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensorRandom.h"
#else

THD_API void THDTensor_(random)(THDTensor *self, THDGenerator *_generator);
THD_API void THDTensor_(geometric)(THDTensor *self, THDGenerator *_generator,
                                   double p);
THD_API void THDTensor_(bernoulli)(THDTensor *self, THDGenerator *_generator,
                                   double p);
THD_API void THDTensor_(bernoulli_FloatTensor)(THDTensor *self,
                                               THDGenerator *_generator,
                                               THDFloatTensor *p);
THD_API void THDTensor_(bernoulli_DoubleTensor)(THDTensor *self,
                                                THDGenerator *_generator,
                                                THDDoubleTensor *p);
THD_API void THDTensor_(uniform)(THDTensor *self, THDGenerator *_generator,
                                 double a, double b);
THD_API void THDTensor_(normal)(THDTensor *self, THDGenerator *_generator,
                                double mean, double stdv);
THD_API void THDTensor_(exponential)(THDTensor *self, THDGenerator *_generator,
                                     double lambda);
THD_API void THDTensor_(cauchy)(THDTensor *self, THDGenerator *_generator,
                                double median, double sigma);
THD_API void THDTensor_(logNormal)(THDTensor *self, THDGenerator *_generator,
                                   double mean, double stdv);
THD_API void THDTensor_(truncatedNormal)(THDTensor *self, THDGenerator *_generator,
                                   double mean, double stdv, double min_val, double max_val);
THD_API void THDTensor_(multinomial)(THDLongTensor *self,
                                     THDGenerator *_generator,
                                     THDTensor *prob_dist,
                                     int n_sample,
                                     int with_replacement);

#endif
