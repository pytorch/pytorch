#include "THGeneral.h"
#include "THRandom.h"

/* The initial seed. */
static unsigned long the_initial_seed;

/* Code for the Mersenne Twister random generator.... */
#define n 624
#define m 397
static int left = 1;
static int initf = 0;
static unsigned long *next;
static unsigned long state[n]; /* the array for the state vector  */
/********************************/

/* For normal distribution */
static double normal_x;
static double normal_y;
static double normal_rho;
static int normal_is_valid = 0;

unsigned long THRandom_seed()
{
  unsigned long s = (unsigned long)time(0);
  THRandom_manualSeed(s);
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

void THRandom_manualSeed(unsigned long the_seed_)
{
  int j;
  the_initial_seed = the_seed_;
  state[0]= the_initial_seed & 0xffffffffUL;
  for(j = 1; j < n; j++)
  {
    state[j] = (1812433253UL * (state[j-1] ^ (state[j-1] >> 30)) + j); 
    /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
    /* In the previous versions, mSBs of the seed affect   */
    /* only mSBs of the array state[].                        */
    /* 2002/01/09 modified by makoto matsumoto             */
    state[j] &= 0xffffffffUL;  /* for >32 bit machines */
  }
  left = 1;
  initf = 1;
}

unsigned long THRandom_initialSeed()
{
  if(initf == 0)
  {
    THRandom_seed();
  }

  return the_initial_seed;
}

void THRandom_nextState()
{
  unsigned long *p=state;
  int j;

  /* if init_genrand() has not been called, */
  /* a default initial seed is used         */
  if(initf == 0)
    THRandom_seed();

  left = n;
  next = state;
    
  for(j = n-m+1; --j; p++) 
    *p = p[m] ^ TWIST(p[0], p[1]);

  for(j = m; --j; p++) 
    *p = p[m-n] ^ TWIST(p[0], p[1]);

  *p = p[m-n] ^ TWIST(p[0], state[0]);
}

unsigned long THRandom_random()
{
  unsigned long y;

  if (--left == 0)
    THRandom_nextState();
  y = *next++;
  
  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);

  return y;
}

/* generates a random number on [0,1)-double-interval */
static double __uniform__()
{
  unsigned long y;

  if(--left == 0)
    THRandom_nextState();
  y = *next++;

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

double THRandom_uniform(double a, double b)
{
  return(__uniform__() * (b - a) + a);
}

double THRandom_normal(double mean, double stdv)
{
  THArgCheck(stdv > 0, 2, "standard deviation must be strictly positive");

  if(!normal_is_valid)
  {
    normal_x = __uniform__();
    normal_y = __uniform__();
    normal_rho = sqrt(-2. * log(1.0-normal_y));
    normal_is_valid = 1;
  }
  else
    normal_is_valid = 0;
  
  if(normal_is_valid)
    return normal_rho*cos(2.*M_PI*normal_x)*stdv+mean;
  else
    return normal_rho*sin(2.*M_PI*normal_x)*stdv+mean;
}

double THRandom_exponential(double lambda)
{
  return(-1. / lambda * log(1-__uniform__()));
}

double THRandom_cauchy(double median, double sigma)
{
  return(median + sigma * tan(M_PI*(__uniform__()-0.5)));
}

/* Faut etre malade pour utiliser ca.
   M'enfin. */
double THRandom_logNormal(double mean, double stdv)
{
  double zm = mean*mean;
  double zs = stdv*stdv;
  THArgCheck(stdv > 0, 2, "standard deviation must be strictly positive");
  return(exp(THRandom_normal(log(zm/sqrt(zs + zm)), sqrt(log(zs/zm+1)) )));
}

int THRandom_geometric(double p)
{
  THArgCheck(p > 0 && p < 1, 1, "must be > 0 and < 1");
  return((int)(log(1-__uniform__()) / log(p)) + 1);
}

int THRandom_bernoulli(double p)
{
  THArgCheck(p >= 0 && p <= 1, 1, "must be >= 0 and <= 1");
  return(__uniform__() <= p);
}
