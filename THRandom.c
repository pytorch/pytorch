#include "THGeneral.h"
#include "THRandom.h"


/* Code for the Mersenne Twister random generator.... */
#define n _MERSENNE_STATE_N
#define m _MERSENNE_STATE_M
THGenerator* THGenerator_new()
{
    mersenne_state *self = THAlloc(sizeof(mersenne_state));
    self->left = 1;
    self->initf = 0;
    self->normal_is_valid = 0;
    return self;
}

void THGenerator_free(THGenerator *self)
{
    THFree(self);
}

unsigned long THRandom_seed(mersenne_state * _mersenne)
{
  unsigned long s = (unsigned long)time(0);
  THRandom_manualSeed(_mersenne, s);
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

void THRandom_manualSeed(mersenne_state * _mersenne, unsigned long the_seed_)
{
  int j;
  _mersenne->the_initial_seed = the_seed_;
  _mersenne->state[0] = _mersenne->the_initial_seed & 0xffffffffUL;
  for(j = 1; j < n; j++)
  {
    _mersenne->state[j] = (1812433253UL * (_mersenne->state[j-1] ^ (_mersenne->state[j-1] >> 30)) + j); 
    /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
    /* In the previous versions, mSBs of the seed affect   */
    /* only mSBs of the array state[].                        */
    /* 2002/01/09 modified by makoto matsumoto             */
    _mersenne->state[j] &= 0xffffffffUL;  /* for >32 bit machines */
  }
  _mersenne->left = 1;
  _mersenne->initf = 1;
}

unsigned long THRandom_initialSeed(mersenne_state * _mersenne)
{
  if(_mersenne->initf == 0)
  {
    THRandom_seed(_mersenne);
  }

  return _mersenne->the_initial_seed;
}

void THRandom_nextState(mersenne_state * _mersenne)
{
  unsigned long *p = _mersenne->state;
  int j;

  /* if init_genrand() has not been called, */
  /* a default initial seed is used         */
  if(_mersenne->initf == 0)
    THRandom_seed(_mersenne);

  _mersenne->left = n;
  _mersenne->next = _mersenne->state;
    
  for(j = n-m+1; --j; p++) 
    *p = p[m] ^ TWIST(p[0], p[1]);

  for(j = m; --j; p++) 
    *p = p[m-n] ^ TWIST(p[0], p[1]);

  *p = p[m-n] ^ TWIST(p[0], p[0]);
}

unsigned long THRandom_random(mersenne_state * _mersenne)
{
  unsigned long y;

  if (--(_mersenne->left) == 0)
    THRandom_nextState(_mersenne);
  y = *((_mersenne->next)++);
  
  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);

  return y;
}

/* generates a random number on [0,1)-double-interval */
static double __uniform__(mersenne_state * _mersenne)
{
  unsigned long y;

  if (--(_mersenne->left) == 0)
    THRandom_nextState(_mersenne);
  y = *((_mersenne->next)++);
  
  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);
  
  return (double)y * (1.0/4294967296.0); 
  /* divided by 2^32 */
}

/*********************************************************

 Thanks *a lot* Takuji Nishimura and Makoto Matsumoto!

 Now my own code...

*********************************************************/

double THRandom_uniform(mersenne_state * _mersenne, double a, double b)
{
  return(__uniform__(_mersenne) * (b - a) + a);
}

double THRandom_normal(mersenne_state * _mersenne, double mean, double stdv)
{
  THArgCheck(stdv > 0, 2, "standard deviation must be strictly positive");

  /* This is known as the Box-Muller method */
  if(!_mersenne->normal_is_valid)
  {
    _mersenne->normal_x = __uniform__(_mersenne);
    _mersenne->normal_y = __uniform__(_mersenne);
    _mersenne->normal_rho = sqrt(-2. * log(1.0-_mersenne->normal_y));
    _mersenne->normal_is_valid = 1;
  }
  else
    _mersenne->normal_is_valid = 0;
  
  if(_mersenne->normal_is_valid)
    return _mersenne->normal_rho*cos(2.*M_PI*_mersenne->normal_x)*stdv+mean;
  else
    return _mersenne->normal_rho*sin(2.*M_PI*_mersenne->normal_x)*stdv+mean;
}

double THRandom_exponential(mersenne_state * _mersenne, double lambda)
{
  return(-1. / lambda * log(1-__uniform__(_mersenne)));
}

double THRandom_cauchy(mersenne_state * _mersenne, double median, double sigma)
{
  return(median + sigma * tan(M_PI*(__uniform__(_mersenne)-0.5)));
}

/* Faut etre malade pour utiliser ca.
   M'enfin. */
double THRandom_logNormal(mersenne_state * _mersenne, double mean, double stdv)
{
  double zm = mean*mean;
  double zs = stdv*stdv;
  THArgCheck(stdv > 0, 2, "standard deviation must be strictly positive");
  return(exp(THRandom_normal(_mersenne, log(zm/sqrt(zs + zm)), sqrt(log(zs/zm+1)) )));
}

int THRandom_geometric(mersenne_state * _mersenne, double p)
{
  THArgCheck(p > 0 && p < 1, 1, "must be > 0 and < 1");
  return((int)(log(1-__uniform__(_mersenne)) / log(p)) + 1);
}

int THRandom_bernoulli(mersenne_state * _mersenne, double p)
{
  THArgCheck(p >= 0 && p <= 1, 1, "must be >= 0 and <= 1");
  return(__uniform__(_mersenne) <= p);
}

/* returns the random number state */
void THRandom_getState(mersenne_state * _mersenne, unsigned long *_state, long *offset, long *_left)
{
  if(_mersenne->initf == 0)
    THRandom_seed(_mersenne);
  memmove(_state, _mersenne->state, n*sizeof(long));
  *offset = (long)(_mersenne->next - _mersenne->state);
  *_left = _mersenne->left;
}

/* sets the random number state */
void THRandom_setState(mersenne_state * _mersenne, unsigned long *_state, long offset, long _left)
{
  memmove(_mersenne->state, _state, n*sizeof(long));
  _mersenne->next = _mersenne->state + offset;
  _mersenne->left = _left;
  _mersenne->initf = 1;
}
