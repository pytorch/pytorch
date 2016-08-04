#ifndef TH_RANDOM_INC
#define TH_RANDOM_INC

#include "THGeneral.h"

#define _MERSENNE_STATE_N 624
#define _MERSENNE_STATE_M 397
/* A THGenerator contains all the state required for a single random number stream */
typedef struct THGenerator {
  /* The initial seed. */
  unsigned long the_initial_seed;
  int left;  /* = 1; */
  int seeded; /* = 0; */
  unsigned long next;
  unsigned long state[_MERSENNE_STATE_N]; /* the array for the state vector  */
  /********************************/

  /* For normal distribution */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid; /* = 0; */
} THGenerator;

#define torch_Generator "torch.Generator"

/* Manipulate THGenerator objects */
TH_API THGenerator * THGenerator_new(void);
TH_API THGenerator * THGenerator_copy(THGenerator *self, THGenerator *from);
TH_API void THGenerator_free(THGenerator *gen);

/* Checks if given generator is valid */
TH_API int THGenerator_isValid(THGenerator *_generator);

/* Initializes the random number generator from /dev/urandom (or on Windows
platforms with the current time (granularity: seconds)) and returns the seed. */
TH_API unsigned long THRandom_seed(THGenerator *_generator);

/* Initializes the random number generator with the given long "the_seed_". */
TH_API void THRandom_manualSeed(THGenerator *_generator, unsigned long the_seed_);

/* Returns the starting seed used. */
TH_API unsigned long THRandom_initialSeed(THGenerator *_generator);

/* Generates a uniform 32 bits integer. */
TH_API unsigned long THRandom_random(THGenerator *_generator);

/* Generates a uniform random number on [0,1[. */
TH_API double THRandom_uniform(THGenerator *_generator, double a, double b);

/** Generates a random number from a normal distribution.
    (With mean #mean# and standard deviation #stdv >= 0#).
*/
TH_API double THRandom_normal(THGenerator *_generator, double mean, double stdv);

/** Generates a random number from an exponential distribution.
    The density is $p(x) = lambda * exp(-lambda * x)$, where
    lambda is a positive number.
*/
TH_API double THRandom_exponential(THGenerator *_generator, double lambda);

/** Returns a random number from a Cauchy distribution.
    The Cauchy density is $p(x) = sigma/(pi*(sigma^2 + (x-median)^2))$
*/
TH_API double THRandom_cauchy(THGenerator *_generator, double median, double sigma);

/** Generates a random number from a log-normal distribution.
    (#mean > 0# is the mean of the log-normal distribution
    and #stdv# is its standard deviation).
*/
TH_API double THRandom_logNormal(THGenerator *_generator, double mean, double stdv);

/** Generates a random number from a geometric distribution.
    It returns an integer #i#, where $p(i) = (1-p) * p^(i-1)$.
    p must satisfy $0 < p < 1$.
*/
TH_API int THRandom_geometric(THGenerator *_generator, double p);

/* Returns true with probability $p$ and false with probability $1-p$ (p > 0). */
TH_API int THRandom_bernoulli(THGenerator *_generator, double p);
#endif
