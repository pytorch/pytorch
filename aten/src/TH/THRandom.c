#include "THGeneral.h"
#include "THRandom.h"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

/* Code for the Mersenne Twister random generator.... */
#define n _MERSENNE_STATE_N
#define m _MERSENNE_STATE_M

/* Creates (unseeded) new generator*/
static THGenerator* THGenerator_newUnseeded()
{
  THGenerator *self = THAlloc(sizeof(THGenerator));
  memset(self, 0, sizeof(THGenerator));
  self->left = 1;
  self->seeded = 0;
  self->normal_is_valid = 0;
  return self;
}

/* Creates new generator and makes sure it is seeded*/
THGenerator* THGenerator_new()
{
  THGenerator *self = THGenerator_newUnseeded();
  THRandom_seed(self);
  return self;
}

THGenerator* THGenerator_copy(THGenerator *self, THGenerator *from)
{
    memcpy(self, from, sizeof(THGenerator));
    return self;
}

void THGenerator_free(THGenerator *self)
{
  THFree(self);
}

int THGenerator_isValid(THGenerator *_generator)
{
  if ((_generator->seeded == 1) &&
    (_generator->left > 0 && _generator->left <= n) && (_generator->next <= n))
    return 1;

  return 0;
}

#ifndef _WIN32
static uint64_t readURandomLong()
{
  int randDev = open("/dev/urandom", O_RDONLY);
  uint64_t randValue;
  if (randDev < 0) {
    THError("Unable to open /dev/urandom");
  }
  ssize_t readBytes = read(randDev, &randValue, sizeof(randValue));
  if (readBytes < sizeof(randValue)) {
    THError("Unable to read from /dev/urandom");
  }
  close(randDev);
  return randValue;
}
#endif // _WIN32

uint64_t THRandom_seed(THGenerator *_generator)
{
#ifdef _WIN32
  uint64_t s = (uint64_t)time(0);
#else
  uint64_t s = readURandomLong();
#endif
  THRandom_manualSeed(_generator, s);
  return s;
}

/* The next 4 methods are taken from http:www.math.keio.ac.jpmatumotoemt.html
   Here is the copyright:
   Some minor modifications have been made to adapt to "my" C... */

/*
   A C-program for MT19937, with initialization improved 2002/2/10.
   Coded by Takuji Nishimura and Makoto Matsumoto.
   This is a faster version by taking Shawn Cokus's optimization,
   Matthe Bellew's simplification, Isaku Wada's double version.

   Before using, initialize the state by using init_genrand(seed)
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote
        products derived from this software without specific prior written
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.keio.ac.jp/matumoto/emt.html
   email: matumoto@math.keio.ac.jp
*/

/* Macros for the Mersenne Twister random generator... */
/* Period parameters */
/* #define n 624 */
/* #define m 397 */
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UMASK 0x80000000UL /* most significant w-r bits */
#define LMASK 0x7fffffffUL /* least significant r bits */
#define MIXBITS(u,v) ( ((u) & UMASK) | ((v) & LMASK) )
#define TWIST(u,v) ((MIXBITS(u,v) >> 1) ^ ((v)&1UL ? MATRIX_A : 0UL))
/*********************************************************** That's it. */

void THRandom_manualSeed(THGenerator *_generator, uint64_t the_seed_)
{
  int j;

  /* This ensures reseeding resets all of the state (i.e. state for Gaussian numbers) */
  THGenerator *blank = THGenerator_newUnseeded();
  THGenerator_copy(_generator, blank);
  THGenerator_free(blank);

  _generator->the_initial_seed = the_seed_;
  _generator->state[0] = _generator->the_initial_seed & 0xffffffffUL;
  for(j = 1; j < n; j++)
  {
    _generator->state[j] = (1812433253UL * (_generator->state[j-1] ^ (_generator->state[j-1] >> 30)) + j);
    /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
    /* In the previous versions, mSBs of the seed affect   */
    /* only mSBs of the array state[].                        */
    /* 2002/01/09 modified by makoto matsumoto             */
    _generator->state[j] &= 0xffffffffUL;  /* for >32 bit machines */
  }
  _generator->left = 1;
  _generator->seeded = 1;
}

uint64_t THRandom_initialSeed(THGenerator *_generator)
{
  return _generator->the_initial_seed;
}

void THRandom_nextState(THGenerator *_generator)
{
  uint64_t *p = _generator->state;
  int j;

  _generator->left = n;
  _generator->next = 0;

  for(j = n-m+1; --j; p++)
    *p = p[m] ^ TWIST(p[0], p[1]);

  for(j = m; --j; p++)
    *p = p[m-n] ^ TWIST(p[0], p[1]);

  *p = p[m-n] ^ TWIST(p[0], _generator->state[0]);
}

// TODO: this only returns 32-bits of randomness but as a uint64_t. This is
// weird and should be fixed. We should also fix the state to be uint32_t
// instead of uint64_t. (Or switch to a 64-bit random number generator).
uint64_t THRandom_random(THGenerator *_generator)
{
  uint64_t y;

  if (--(_generator->left) == 0)
    THRandom_nextState(_generator);
  y = *(_generator->state + (_generator->next)++);

  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);

  return y;
}

uint64_t THRandom_random64(THGenerator *_generator)
{
  uint64_t hi = THRandom_random(_generator);
  uint64_t lo = THRandom_random(_generator);
  return (hi << 32) | lo;
}

// doubles have 52 bits of mantissa (fractional part)
static uint64_t DOUBLE_MASK = (1ULL << 53) - 1;
static double DOUBLE_DIVISOR = 1.0 / (1ULL << 53);

// floats have 23 bits of mantissa (fractional part)
static uint32_t FLOAT_MASK = (1 << 24) - 1;
static float FLOAT_DIVISOR = 1.0f / (1 << 24);

/* generates a random number on [0,1)-double-interval */
static double uniform_double(THGenerator *_generator)
{
  uint64_t x = THRandom_random64(_generator);
  return (x & DOUBLE_MASK) * DOUBLE_DIVISOR;
}

/* generates a random number on [0,1)-double-interval */
static float uniform_float(THGenerator *_generator)
{
  uint32_t x = (uint32_t)THRandom_random(_generator);
  return (x & FLOAT_MASK) * FLOAT_DIVISOR;
}

/*********************************************************

 Thanks *a lot* Takuji Nishimura and Makoto Matsumoto!

 Now my own code...

*********************************************************/

double THRandom_uniform(THGenerator *_generator, double a, double b)
{
  return(uniform_double(_generator) * (b - a) + a);
}

float THRandom_uniformFloat(THGenerator *_generator, float a, float b)
{
  return(uniform_float(_generator) * (b - a) + a);
}

double THRandom_normal(THGenerator *_generator, double mean, double stdv)
{
  THArgCheck(stdv > 0, 2, "standard deviation must be strictly positive");

  /* This is known as the Box-Muller method */
  if(!_generator->normal_is_valid)
  {
    _generator->normal_x = uniform_double(_generator);
    _generator->normal_y = uniform_double(_generator);
    _generator->normal_rho = sqrt(-2. * log(1.0-_generator->normal_y));
    _generator->normal_is_valid = 1;
  }
  else
    _generator->normal_is_valid = 0;

  if(_generator->normal_is_valid)
    return _generator->normal_rho*cos(2.*M_PI*_generator->normal_x)*stdv+mean;
  else
    return _generator->normal_rho*sin(2.*M_PI*_generator->normal_x)*stdv+mean;
}

double THRandom_exponential(THGenerator *_generator, double lambda)
{
  return(-1. / lambda * log(1-uniform_double(_generator)));
}

double THRandom_standard_gamma(THGenerator *_generator, double alpha) {
  double scale = 1.0;

  // Boost alpha for higher acceptance probability.
  if(alpha < 1.0) {
    scale *= pow(1 - uniform_double(_generator), 1.0 / alpha);
    alpha += 1.0;
  }

  // This implements the acceptance-rejection method of Marsaglia and Tsang (2000)
  // doi:10.1145/358407.358414
  const double d = alpha - 1.0 / 3.0;
  const double c = 1.0 / sqrt(9.0 * d);
  for(;;) {
    double x, y;
    do {
      x = THRandom_normal(_generator, 0.0, 1.0);
      y = 1.0 + c * x;
    } while(y <= 0);
    const double v = y * y * y;
    const double u = 1 - uniform_double(_generator);
    const double xx = x * x;
    if(u < 1.0 - 0.0331 * xx * xx)
      return scale * d * v;
    if(log(u) < 0.5 * xx + d * (1.0 - v + log(v)))
      return scale * d * v;
  }
}

// TODO Replace this with more accurate digamma().
static inline double _digamma(double x) {
  const double eps = x * 1e-2;
  return (lgamma(x + eps) - lgamma(x - eps)) / (eps + eps);
}

// Approximate reparameterized gradient of Beta(x,alpha,beta) wrt alpha.
// Assumes x is close to zero.
static inline double beta_grad_alpha_small(double x, double alpha, double beta) {
  const double b1 = beta - 1.0;
  const double b2 = beta - 2.0;
  const double b3 = beta - 3.0;
  const double a0 = 1.0 / alpha;
  const double a1 = 1.0 / (alpha + 1.0);
  const double a2 = 1.0 / (alpha + 2.0);
  const double a3 = 1.0 / (alpha + 3.0);
  // Let pdf = pow(x,alpha-1) * pow(1-x,beta-1) / Beta(alpha,beta).
  // Let const = Beta(alpha,beta) / pow(x, alpha). Then
  const double one_over_const_pdf = x / pow(1 - x, beta - 1);
  const double const_cdf = +a0 + b1 * x * (
                           -a1 + b2 * x / 2 * (
                           +a2 + b3 * x / 3 * (
                           -a3)));
  const double const_cdf_alpha = (log(x) + _digamma(alpha + beta) - _digamma(alpha)) * const_cdf
        + -a0 * a0 + b1 * x * (
          +a1 * a1 + b2 * x / 2 * (
          -a2 * a2 + b3 * x / 3 * (
          +a3)));
  const double result = -const_cdf_alpha * one_over_const_pdf;
  return isnan(result) ? 0.0 : result;
}

// Approximate reparameterized gradient of Beta(x,alpha,beta) wrt beta.
// Assumes x is close to zero.
static inline double beta_grad_beta_small(double x, double alpha, double beta) {
  const double a0 = 1.0 / alpha;
  const double a1 = 1.0 / (alpha + 1.0);
  const double a2 = 1.0 / (alpha + 2.0);
  const double a3 = 1.0 / (alpha + 3.0);
  // Let pdf = pow(x,alpha-1) * pow(1-x,beta-1) / Beta(alpha,beta).
  // Let const = Beta(alpha,beta) / pow(x, alpha). Then
  const double one_over_const_pdf = x / pow(1 - x, beta - 1);
  const double const_cdf = +a0 + (beta - 1.0) * x * (
                           -a1 + (beta - 2.0) * x / 2 * (
                           +a2 + (beta - 3.0) * x / 3 * (
                           -a3)));
  const double const_cdf_beta = (_digamma(alpha + beta) - _digamma(beta)) * const_cdf
                               + 0 + x * (
                               -a1 + x / 2 * (
                               +a2 * (2 * beta - 3) + x / 3 * (
                               -a3 * (3 * beta * beta - 12 * beta + 11))));
  const double result = -const_cdf_beta * one_over_const_pdf;
  return isnan(result) ? 0.0 : result;
}

double THRandom_dirichlet_grad(double x, double alpha, double total) {
  const double beta = total - alpha;

  // Use an asymptotic approximation for x close to 0.
  if (x * (1.0 + total) < 0.75) {
    return beta_grad_alpha_small(x, alpha, beta);
  }

  // Use an asymptotic approximation for x close to 1.
  if ((1.0 - x) * (1.0 + total) < 0.5) {
    return -beta_grad_beta_small(1.0 - x, beta, alpha);
  }

  // Use a Laplace approximation when alpha and (total - alpha) are both large.
  if (alpha > 100 && beta > 100) {
    const double logit = x / (1.0 - x);
    const double Logit = alpha / beta;
    return x * (1 - x) * (1.0 / alpha - log(logit / Logit) / (2 * Logit * total));
  }

  // Use a rational correction to an analytic baseline.
  static const double c[2][3][3][3] = {
    {{{0.9725276563, -0.0509239565, 1.625070847e-06},
      {0.03797015233, 0.007409446855, -0.0008465634691},
      {0.04498118832, -0.005657730635, 0.0006594371985}},
     {{-0.4906583778, -0.03501797976, 0.0116832438},
      {0.07144210068, 0.01917836117, -0.01049408328},
      {-0.01850287138, 0.0009313017859, 0.001438917764}},
     {{-0.04252776008, 0.07983672436, -0.005747111986},
      {0.01119087258, -0.03621480878, 0.002405820172},
      {0.008194427563, 0.002768270684, -0.0001137741838}}},
    {{{1, -0.06216981624, -0.001621712526},
      {0.03395698453, 0.01103224173, -0.000539324895},
      {0.0438507131, -0.005479208201, 0.000601035911}},
     {{0.007131342965, 0.03959558295, -0.003737549484},
      {0.02450054108, 0.001429734367, -0.0002356644527},
      {-0.004480783661, 0.004028770843, -0.0002935849718}},
     {{-0.0046942397, 0.003926730136, -0.0003397299665},
      {0.009286906003, -0.001605160177, 7.69956057e-05},
      {-0.0006000287485, 0.0004877957422, -4.36865244e-05}}},
  };
  const double u = log(x);
  const double a = log(alpha);
  const double b = log(total);
  const double us[4] = {1.0, u, u * u, u * u * u};
  const double as[4] = {1.0, a, a * a, a * a * a};
  double p = 0.0;
  double q = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double ua = us[i] * as[j];
      p += ua * (c[0][i][j][0] + b * (c[0][i][j][1] + b * c[0][i][j][2]));
      q += ua * (c[1][i][j][0] + b * (c[1][i][j][1] + b * c[1][i][j][2]));
    }
  }
  if(q < 1e-3) q = 1e-3;
  return p / q * x * (1.0 - x) * (_digamma(total) - _digamma(alpha)) / beta;
}

double THRandom_cauchy(THGenerator *_generator, double median, double sigma)
{
  return(median + sigma * tan(M_PI*(uniform_double(_generator)-0.5)));
}

/* Faut etre malade pour utiliser ca.
   M'enfin. */
double THRandom_logNormal(THGenerator *_generator, double mean, double stdv)
{
  THArgCheck(stdv > 0, 2, "standard deviation must be strictly positive");
  return(exp(THRandom_normal(_generator, mean, stdv)));
}

int THRandom_geometric(THGenerator *_generator, double p)
{
  THArgCheck(p > 0 && p < 1, 1, "must be > 0 and < 1");
  return((int)(log(1-uniform_double(_generator)) / log(p)) + 1);
}

int THRandom_bernoulli(THGenerator *_generator, double p)
{
  THArgCheck(p >= 0 && p <= 1, 1, "must be >= 0 and <= 1");
  return(uniform_double(_generator) <= p);
}
