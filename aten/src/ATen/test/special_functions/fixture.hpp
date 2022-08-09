#ifndef FIXTURE_HPP
#define FIXTURE_HPP

#include <complex>

template<typename T1>
struct testcase_coulomb_f {
  T1 f0;
  T1 lambda;
  T1 eta;
  T1 rho;
  T1 f;
};

template<typename T1>
struct testcase_coulomb_g {
  T1 f0;
  T1 lambda;
  T1 eta;
  T1 rho;
  T1 f;
};

template<typename T1>
struct bessel_fixture {
  T1 f0;
  T1 n;
  T1 x;
  T1 f;
};

template<typename T1>
struct spherical_bessel_fixture {
  T1 f0;
  unsigned int n;
  T1 x;
  T1 f;
};

template<typename T1>
struct testcase_gamma_q {
  T1 f0;
  T1 a;
  T1 x;
  T1 f;
};

template<typename T1>
struct testcase_gamma_p {
  T1 f0;
  T1 a;
  T1 x;
  T1 f;
};

template<typename T1>
struct gamma_fixture {
  T1 f0;
  T1 x;
  T1 f;
};

template<typename T1>
struct testcase_comp_ellint_rg {
  T1 f0;
  T1 x;
  T1 y;
  T1 f;
};

template<typename T1>
struct testcase_comp_ellint_rf {
  T1 f0;
  T1 x;
  T1 y;
  T1 f;
};

template<typename T1>
struct testcase_ellint_rj {
  T1 f0;
  T1 x;
  T1 y;
  T1 z;
  T1 p;
  T1 f;
};

template<typename T1>
struct testcase_dawson {
  T1 f0;
  T1 x;
  T1 f;
};

template<typename T1>
struct jacobi_elliptic_sn_fixture {
  T1 f0;
  T1 k;
  T1 u;
  T1 f;
};

template<typename T1>
struct jacobi_elliptic_cn_fixture {
  T1 f0;
  T1 k;
  T1 u;
  T1 f;
};

template<typename T1>
struct jacobi_elliptic_dn_fixture {
  T1 f0;
  T1 k;
  T1 u;
  T1 f;
};

template<typename T1>
struct sinc_fixture {
  T1 f0;
  T1 x;
  T1 f;
};

template<typename T1>
struct testcase_ibetac {
  T1 f0;
  T1 a;
  T1 b;
  T1 x;
  T1 f;
};



template<typename T1>
struct testcase_polylog {
  std::complex<T1> f0;
  T1 s;
  std::complex<T1> w;
  std::complex<T1> f;
};

template<typename T1>
struct testcase_clausen {
  std::complex<T1> f0;
  unsigned int m;
  std::complex<T1> w;
  std::complex<T1> f;
};

template<typename T1>
struct tan_pi_fixture {
  T1 f0;
  T1 x;
  T1 f;
};

//
template<typename T1>
struct testcase_boltzmann_p {
  T1 f0;
  T1 mu;
  T1 x;
  T1 f;
};

//
template<typename T1>
struct testcase_boltzmann_pdf {
  T1 f0;
  T1 mu;
  T1 x;
  T1 f;
};

//
template<typename T1>
struct testcase_laplace_p {
  T1 f0;
  T1 mu;
  T1 sigma;
  T1 x;
  T1 f;
};

//
template<typename T1>
struct testcase_laplace_pdf {
  T1 f0;
  T1 mu;
  T1 sigma;
  T1 x;
  T1 f;
};

//
template<typename T1>
struct testcase_maxwell_p {
  T1 f0;
  T1 mu;
  T1 f;
};

//
template<typename T1>
struct testcase_maxwell_pdf {
  T1 f0;
  T1 mu;
  T1 f;
};

//
template<typename T1>
struct testcase_normal_p {
  T1 f0;
  T1 mu;
  T1 sigma;
  T1 x;
  T1 f;
};

//
template<typename T1>
struct testcase_normal_pdf {
  T1 f0;
  T1 mu;
  T1 sigma;
  T1 x;
  T1 f;
};

//
template<typename T1>
struct testcase_rayleigh_p {
  T1 f0;
  T1 mu;
  T1 x;
  T1 f;
};

//
template<typename T1>
struct testcase_rayleigh_pdf {
  T1 f0;
  T1 mu;
  T1 x;
  T1 f;
};

//
template<typename T1>
struct testcase_lognormal_p {
  T1 f0;
  T1 mu;
  T1 sigma;
  T1 x;
  T1 f;
};

//
template<typename T1>
struct testcase_lognormal_pdf {
  T1 f0;
  T1 mu;
  T1 sigma;
  T1 x;
  T1 f;
};

//
template<typename T1>
struct testcase_logistic_p {
  T1 f0;
  T1 mu;
  T1 s;
  T1 x;
  T1 f;
};

//
template<typename T1>
struct testcase_logistic_pdf {
  T1 f0;
  T1 mu;
  T1 s;
  T1 x;
  T1 f;
};

template<typename T1>
struct testcase_euler {
  T1 f0;
  unsigned int n;
  T1 f;
};

template<typename T1>
struct testcase_eulerian_1 {
  T1 f0;
  unsigned int n;
  unsigned int m;
  T1 f;
};

template<typename T1>
struct testcase_eulerian_2 {
  T1 f0;
  unsigned int n;
  unsigned int m;
  T1 f;
};

#endif
