#ifndef TH_RANDOM_INC
#define TH_RANDOM_INC

#include "THGeneral.h"

#define _MERSENNE_STATE_N 624 
#define _MERSENNE_STATE_M 397
typedef struct mersenne_state {
  /* The initial seed. */
  unsigned long the_initial_seed;
  int left;  /* = 1; */
  int initf; /* = 0; */
  unsigned long *next;
  unsigned long state[_MERSENNE_STATE_N]; /* the array for the state vector  */
  /********************************/
  
  /* For normal distribution */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid; /* = 0; */
} mersenne_state;

#define THRandomGenerator mersenne_state
#define THGenerator mersenne_state
#define torch_Generator "torch.Generator"

/* Initializes the random number generator with the current time (granularity: seconds) and returns the seed. */
TH_API unsigned long THRandom_seed(mersenne_state * _mersenne);

/* Initializes the random number generator with the given long "the_seed_". */
TH_API void THRandom_manualSeed(mersenne_state * _mersenne, unsigned long the_seed_);

/* Returns the starting seed used. */
TH_API unsigned long THRandom_initialSeed(mersenne_state * _mersenne);

/* Generates a uniform 32 bits integer. */
TH_API unsigned long THRandom_random(mersenne_state * _mersenne);

/* Generates a uniform random number on [0,1[. */
TH_API double THRandom_uniform(mersenne_state * _mersenne, double a, double b);

/** Generates a random number from a normal distribution.
    (With mean #mean# and standard deviation #stdv >= 0#).
*/
TH_API double THRandom_normal(mersenne_state * _mersenne, double mean, double stdv);

/** Generates a random number from an exponential distribution.
    The density is $p(x) = lambda * exp(-lambda * x)$, where
    lambda is a positive number.
*/
TH_API double THRandom_exponential(mersenne_state * _mersenne, double lambda);

/** Returns a random number from a Cauchy distribution.
    The Cauchy density is $p(x) = sigma/(pi*(sigma^2 + (x-median)^2))$
*/
TH_API double THRandom_cauchy(mersenne_state * _mersenne, double median, double sigma);

/** Generates a random number from a log-normal distribution.
    (#mean > 0# is the mean of the log-normal distribution
    and #stdv# is its standard deviation).
*/
TH_API double THRandom_logNormal(mersenne_state * _mersenne, double mean, double stdv);

/** Generates a random number from a geometric distribution.
    It returns an integer #i#, where $p(i) = (1-p) * p^(i-1)$.
    p must satisfy $0 < p < 1$.
*/
TH_API int THRandom_geometric(mersenne_state * _mersenne, double p);

/* Returns true with probability $p$ and false with probability $1-p$ (p > 0). */
TH_API int THRandom_bernoulli(mersenne_state * _mersenne, double p);

/* returns the random number state */
TH_API void THRandom_getState(mersenne_state * _mersenne, unsigned long *state, long *offset, long *_left);

/* sets the random number state */
TH_API void THRandom_setState(mersenne_state * _mersenne, unsigned long *state, long offset, long _left);
#endif
