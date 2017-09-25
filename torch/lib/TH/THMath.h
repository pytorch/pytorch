#ifndef _THMATH_H
#define _THMATH_H
#include <stdlib.h>
#include <math.h>

static inline double TH_sigmoid(double value) {
  return 1.0 / (1.0 + exp(-value));
}

static inline double TH_frac(double x) {
  return x - trunc(x);
}

static inline double TH_rsqrt(double x) {
  return 1.0 / sqrt(x);
}

static inline double TH_lerp(double a, double b, double weight) {
  return a + weight * (b-a);
}

static inline float TH_sigmoidf(float value) {
  return 1.0f / (1.0f + expf(-value));
}

static inline float TH_fracf(float x) {
  return x - truncf(x);
}

static inline float TH_rsqrtf(float x) {
  return 1.0f / sqrtf(x);
}

static inline float TH_lerpf(float a, float b, float weight) {
  return a + weight * (b-a);
}


/* The next function is taken from  https://github.com/antelopeusersgroup/antelope_contrib/blob/master/lib/location/libgenloc/erfinv.c.
Below is the copyright.
Output was modified to be inf or -inf when input is 1 or -1. */


/*
    Copyright (c) 2014 Indiana University
    All rights reserved.

    Written by Prof. Gary L. Pavlis, Dept. of Geol. Sci.,
            Indiana University, Bloomington, IN

    This software is licensed under the New BSD license:

    Redistribution and use in source and binary forms,
    with or without modification, are permitted provided
    that the following conditions are met:

    Redistributions of source code must retain the above
    copyright notice, this list of conditions and the
    following disclaimer.

    Redistributions in binary form must reproduce the
    above copyright notice, this list of conditions and
    the following disclaimer in the documentation and/or
    other materials provided with the distribution.

    Neither the name of Indiana University nor
    the names of its contributors may be used to endorse
    or promote products derived from this software without
    specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
    CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
    WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
    THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
    USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
    IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
    USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
*/

#define CENTRAL_RANGE 0.7

static inline double TH_erfinv(double y) {
/* Function to calculate inverse error function.  Rational approximation
is used to generate an initial approximation, which is then improved to
full accuracy by two steps of Newton's method.  Code is a direct
translation of the erfinv m file in matlab version 2.0.
Author:  Gary L. Pavlis, Indiana University
Date:  February 1996
*/
    double x,z,num,dem; /*working variables */
    /* coefficients in rational expansion */
    double a[4]={ 0.886226899, -1.645349621,  0.914624893, -0.140543331};
    double b[4]={-2.118377725,  1.442710462, -0.329097515,  0.012229801};
    double c[4]={-1.970840454, -1.624906493,  3.429567803,  1.641345311};
    double d[2]={ 3.543889200,  1.637067800};
    if(fabs(y) > 1.0) return (atof("NaN"));  /* This needs IEEE constant*/
    if(fabs(y) == 1.0) return((copysign(1.0,y))*atof("INFINITY"));
    if(fabs(y) <= CENTRAL_RANGE){
            z = y*y;
            num = (((a[3]*z + a[2])*z + a[1])*z + a[0]);
            dem = ((((b[3]*z + b[2])*z + b[1])*z +b[0])*z + 1.0);
            x = y*num/dem;
    }
    else{
            z = sqrt(-log((1.0-fabs(y))/2.0));
            num = ((c[3]*z + c[2])*z + c[1])*z + c[0];
            dem = (d[1]*z + d[0])*z + 1.0;
            x = (copysign(1.0,y))*num/dem;
    }
    /* Two steps of Newton-Raphson correction */
    x = x - (erf(x) - y)/( (2.0/sqrt(M_PI))*exp(-x*x));
    x = x - (erf(x) - y)/( (2.0/sqrt(M_PI))*exp(-x*x));

    return(x);
}
#undef CENTRAL_RANGE

#endif // _THMATH_H
