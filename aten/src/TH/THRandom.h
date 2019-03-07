#ifndef TH_RANDOM_INC
#define TH_RANDOM_INC

#include <TH/THGeneral.h>

#define _MERSENNE_STATE_N 624
#define _MERSENNE_STATE_M 397

/* Struct definition is moved to THGenerator.hpp, because THRandom.h
needs to be C-compatible in order to be included in C FFI extensions. */
typedef struct THGenerator THGenerator;
typedef struct THGeneratorState THGeneratorState;

#define torch_Generator "torch.Generator"

/* Manipulate THGenerator objects */
TH_API THGenerator * THGenerator_new(void);
TH_API THGenerator * THGenerator_copy(THGenerator *self, THGenerator *from);
TH_API void THGenerator_free(THGenerator *gen);

/* Checks if given generator state is valid */
TH_API int THGeneratorState_isValid(THGeneratorState *_gen_state);

/* Manipulate THGeneratorState objects */
TH_API THGeneratorState * THGeneratorState_copy(THGeneratorState *self, THGeneratorState *from);

/* Initializes the random number generator from /dev/urandom (or on Windows
platforms with the current time (granularity: seconds)) and returns the seed. */
TH_API uint64_t THRandom_seed(THGenerator *_generator);

/* Initializes the random number generator with the given int64_t "the_seed_". */
TH_API void THRandom_manualSeed(THGenerator *_generator, uint64_t the_seed_);

/* Returns the starting seed used. */
TH_API uint64_t THRandom_initialSeed(THGenerator *_generator);

/* Generates a uniform 32 bits integer. */
TH_API uint64_t THRandom_random(THGenerator *_generator);

/* Generates a uniform 64 bits integer. */
TH_API uint64_t THRandom_random64(THGenerator *_generator);

/* Generates a uniform random double on [0,1). */
TH_API double THRandom_standard_uniform(THGenerator *_generator);

/* Generates a uniform random double on [a, b). */
TH_API double THRandom_uniform(THGenerator *_generator, double a, double b);

/* Generates a uniform random float on [0,1). */
TH_API float THRandom_uniformFloat(THGenerator *_generator, float a, float b);

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

/* Returns true with double probability $p$ and false with probability $1-p$ (p > 0). */
TH_API int THRandom_bernoulli(THGenerator *_generator, double p);

/* Returns true with float probability $p$ and false with probability $1-p$ (p > 0). */
TH_API int THRandom_bernoulliFloat(THGenerator *_generator, float p);

#endif
