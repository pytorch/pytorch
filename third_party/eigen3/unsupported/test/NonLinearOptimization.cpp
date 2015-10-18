// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>

#include <stdio.h>

#include "main.h"
#include <unsupported/Eigen/NonLinearOptimization>

// This disables some useless Warnings on MSVC.
// It is intended to be done for this test only.
#include <Eigen/src/Core/util/DisableStupidWarnings.h>

using std::sqrt;

int fcn_chkder(const VectorXd &x, VectorXd &fvec, MatrixXd &fjac, int iflag)
{
    /*      subroutine fcn for chkder example. */

    int i;
    assert(15 ==  fvec.size());
    assert(3 ==  x.size());
    double tmp1, tmp2, tmp3, tmp4;
    static const double y[15]={1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
        3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};


    if (iflag == 0)
        return 0;

    if (iflag != 2)
        for (i=0; i<15; i++) {
            tmp1 = i+1;
            tmp2 = 16-i-1;
            tmp3 = tmp1;
            if (i >= 8) tmp3 = tmp2;
            fvec[i] = y[i] - (x[0] + tmp1/(x[1]*tmp2 + x[2]*tmp3));
        }
    else {
        for (i = 0; i < 15; i++) {
            tmp1 = i+1;
            tmp2 = 16-i-1;

            /* error introduced into next statement for illustration. */
            /* corrected statement should read    tmp3 = tmp1 . */

            tmp3 = tmp2;
            if (i >= 8) tmp3 = tmp2;
            tmp4 = (x[1]*tmp2 + x[2]*tmp3); tmp4=tmp4*tmp4;
            fjac(i,0) = -1.;
            fjac(i,1) = tmp1*tmp2/tmp4;
            fjac(i,2) = tmp1*tmp3/tmp4;
        }
    }
    return 0;
}


void testChkder()
{
  const int m=15, n=3;
  VectorXd x(n), fvec(m), xp, fvecp(m), err;
  MatrixXd fjac(m,n);
  VectorXi ipvt;

  /*      the following values should be suitable for */
  /*      checking the jacobian matrix. */
  x << 9.2e-1, 1.3e-1, 5.4e-1;

  internal::chkder(x, fvec, fjac, xp, fvecp, 1, err);
  fcn_chkder(x, fvec, fjac, 1);
  fcn_chkder(x, fvec, fjac, 2);
  fcn_chkder(xp, fvecp, fjac, 1);
  internal::chkder(x, fvec, fjac, xp, fvecp, 2, err);

  fvecp -= fvec;

  // check those
  VectorXd fvec_ref(m), fvecp_ref(m), err_ref(m);
  fvec_ref <<
      -1.181606, -1.429655, -1.606344,
      -1.745269, -1.840654, -1.921586,
      -1.984141, -2.022537, -2.468977,
      -2.827562, -3.473582, -4.437612,
      -6.047662, -9.267761, -18.91806;
  fvecp_ref <<
      -7.724666e-09, -3.432406e-09, -2.034843e-10,
      2.313685e-09,  4.331078e-09,  5.984096e-09,
      7.363281e-09,   8.53147e-09,  1.488591e-08,
      2.33585e-08,  3.522012e-08,  5.301255e-08,
      8.26666e-08,  1.419747e-07,   3.19899e-07;
  err_ref <<
      0.1141397,  0.09943516,  0.09674474,
      0.09980447,  0.1073116, 0.1220445,
      0.1526814, 1, 1,
      1, 1, 1,
      1, 1, 1;

  VERIFY_IS_APPROX(fvec, fvec_ref);
  VERIFY_IS_APPROX(fvecp, fvecp_ref);
  VERIFY_IS_APPROX(err, err_ref);
}

// Generic functor
template<typename _Scalar, int NX=Dynamic, int NY=Dynamic>
struct Functor
{
  typedef _Scalar Scalar;
  enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
  };
  typedef Matrix<Scalar,InputsAtCompileTime,1> InputType;
  typedef Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
  typedef Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

  const int m_inputs, m_values;

  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

  // you should define that in the subclass :
//  void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
};

struct lmder_functor : Functor<double>
{
    lmder_functor(void): Functor<double>(3,15) {}
    int operator()(const VectorXd &x, VectorXd &fvec) const
    {
        double tmp1, tmp2, tmp3;
        static const double y[15] = {1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
            3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};

        for (int i = 0; i < values(); i++)
        {
            tmp1 = i+1;
            tmp2 = 16 - i - 1;
            tmp3 = (i>=8)? tmp2 : tmp1;
            fvec[i] = y[i] - (x[0] + tmp1/(x[1]*tmp2 + x[2]*tmp3));
        }
        return 0;
    }

    int df(const VectorXd &x, MatrixXd &fjac) const
    {
        double tmp1, tmp2, tmp3, tmp4;
        for (int i = 0; i < values(); i++)
        {
            tmp1 = i+1;
            tmp2 = 16 - i - 1;
            tmp3 = (i>=8)? tmp2 : tmp1;
            tmp4 = (x[1]*tmp2 + x[2]*tmp3); tmp4 = tmp4*tmp4;
            fjac(i,0) = -1;
            fjac(i,1) = tmp1*tmp2/tmp4;
            fjac(i,2) = tmp1*tmp3/tmp4;
        }
        return 0;
    }
};

void testLmder1()
{
  int n=3, info;

  VectorXd x;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmder_functor functor;
  LevenbergMarquardt<lmder_functor> lm(functor);
  info = lm.lmder1(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 6);
  VERIFY_IS_EQUAL(lm.njev, 5);

  // check norm
  VERIFY_IS_APPROX(lm.fvec.blueNorm(), 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  VERIFY_IS_APPROX(x, x_ref);
}

void testLmder()
{
  const int m=15, n=3;
  int info;
  double fnorm, covfac;
  VectorXd x;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmder_functor functor;
  LevenbergMarquardt<lmder_functor> lm(functor);
  info = lm.minimize(x);

  // check return values
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 6);
  VERIFY_IS_EQUAL(lm.njev, 5);

  // check norm
  fnorm = lm.fvec.blueNorm();
  VERIFY_IS_APPROX(fnorm, 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  VERIFY_IS_APPROX(x, x_ref);

  // check covariance
  covfac = fnorm*fnorm/(m-n);
  internal::covar(lm.fjac, lm.permutation.indices()); // TODO : move this as a function of lm

  MatrixXd cov_ref(n,n);
  cov_ref <<
      0.0001531202,   0.002869941,  -0.002656662,
      0.002869941,    0.09480935,   -0.09098995,
      -0.002656662,   -0.09098995,    0.08778727;

//  std::cout << fjac*covfac << std::endl;

  MatrixXd cov;
  cov =  covfac*lm.fjac.topLeftCorner<n,n>();
  VERIFY_IS_APPROX( cov, cov_ref);
  // TODO: why isn't this allowed ? :
  // VERIFY_IS_APPROX( covfac*fjac.topLeftCorner<n,n>() , cov_ref);
}

struct hybrj_functor : Functor<double>
{
    hybrj_functor(void) : Functor<double>(9,9) {}

    int operator()(const VectorXd &x, VectorXd &fvec)
    {
        double temp, temp1, temp2;
        const VectorXd::Index n = x.size();
        assert(fvec.size()==n);
        for (VectorXd::Index k = 0; k < n; k++)
        {
            temp = (3. - 2.*x[k])*x[k];
            temp1 = 0.;
            if (k) temp1 = x[k-1];
            temp2 = 0.;
            if (k != n-1) temp2 = x[k+1];
            fvec[k] = temp - temp1 - 2.*temp2 + 1.;
        }
        return 0;
    }
    int df(const VectorXd &x, MatrixXd &fjac)
    {
        const VectorXd::Index n = x.size();
        assert(fjac.rows()==n);
        assert(fjac.cols()==n);
        for (VectorXd::Index k = 0; k < n; k++)
        {
            for (VectorXd::Index j = 0; j < n; j++)
                fjac(k,j) = 0.;
            fjac(k,k) = 3.- 4.*x[k];
            if (k) fjac(k,k-1) = -1.;
            if (k != n-1) fjac(k,k+1) = -2.;
        }
        return 0;
    }
};


void testHybrj1()
{
  const int n=9;
  int info;
  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, -1.);

  // do the computation
  hybrj_functor functor;
  HybridNonLinearSolver<hybrj_functor> solver(functor);
  info = solver.hybrj1(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(solver.nfev, 11);
  VERIFY_IS_EQUAL(solver.njev, 1);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);


// check x
  VectorXd x_ref(n);
  x_ref <<
     -0.5706545,    -0.6816283,    -0.7017325,
     -0.7042129,     -0.701369,    -0.6918656,
     -0.665792,    -0.5960342,    -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}

void testHybrj()
{
  const int n=9;
  int info;
  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, -1.);


  // do the computation
  hybrj_functor functor;
  HybridNonLinearSolver<hybrj_functor> solver(functor);
  solver.diag.setConstant(n, 1.);
  solver.useExternalScaling = true;
  info = solver.solve(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(solver.nfev, 11);
  VERIFY_IS_EQUAL(solver.njev, 1);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);


// check x
  VectorXd x_ref(n);
  x_ref <<
     -0.5706545,    -0.6816283,    -0.7017325,
     -0.7042129,     -0.701369,    -0.6918656,
     -0.665792,    -0.5960342,    -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);

}

struct hybrd_functor : Functor<double>
{
    hybrd_functor(void) : Functor<double>(9,9) {}
    int operator()(const VectorXd &x, VectorXd &fvec) const
    {
        double temp, temp1, temp2;
        const VectorXd::Index n = x.size();

        assert(fvec.size()==n);
        for (VectorXd::Index k=0; k < n; k++)
        {
            temp = (3. - 2.*x[k])*x[k];
            temp1 = 0.;
            if (k) temp1 = x[k-1];
            temp2 = 0.;
            if (k != n-1) temp2 = x[k+1];
            fvec[k] = temp - temp1 - 2.*temp2 + 1.;
        }
        return 0;
    }
};

void testHybrd1()
{
  int n=9, info;
  VectorXd x(n);

  /* the following starting values provide a rough solution. */
  x.setConstant(n, -1.);

  // do the computation
  hybrd_functor functor;
  HybridNonLinearSolver<hybrd_functor> solver(functor);
  info = solver.hybrd1(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(solver.nfev, 20);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);

  // check x
  VectorXd x_ref(n);
  x_ref << -0.5706545, -0.6816283, -0.7017325, -0.7042129, -0.701369, -0.6918656, -0.665792, -0.5960342, -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}

void testHybrd()
{
  const int n=9;
  int info;
  VectorXd x;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, -1.);

  // do the computation
  hybrd_functor functor;
  HybridNonLinearSolver<hybrd_functor> solver(functor);
  solver.parameters.nb_of_subdiagonals = 1;
  solver.parameters.nb_of_superdiagonals = 1;
  solver.diag.setConstant(n, 1.);
  solver.useExternalScaling = true;
  info = solver.solveNumericalDiff(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(solver.nfev, 14);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);

  // check x
  VectorXd x_ref(n);
  x_ref <<
      -0.5706545,    -0.6816283,    -0.7017325,
      -0.7042129,     -0.701369,    -0.6918656,
      -0.665792,    -0.5960342,    -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}

struct lmstr_functor : Functor<double>
{
    lmstr_functor(void) : Functor<double>(3,15) {}
    int operator()(const VectorXd &x, VectorXd &fvec)
    {
        /*  subroutine fcn for lmstr1 example. */
        double tmp1, tmp2, tmp3;
        static const double y[15]={1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
            3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};

        assert(15==fvec.size());
        assert(3==x.size());

        for (int i=0; i<15; i++)
        {
            tmp1 = i+1;
            tmp2 = 16 - i - 1;
            tmp3 = (i>=8)? tmp2 : tmp1;
            fvec[i] = y[i] - (x[0] + tmp1/(x[1]*tmp2 + x[2]*tmp3));
        }
        return 0;
    }
    int df(const VectorXd &x, VectorXd &jac_row, VectorXd::Index rownb)
    {
        assert(x.size()==3);
        assert(jac_row.size()==x.size());
        double tmp1, tmp2, tmp3, tmp4;

        VectorXd::Index i = rownb-2;
        tmp1 = i+1;
        tmp2 = 16 - i - 1;
        tmp3 = (i>=8)? tmp2 : tmp1;
        tmp4 = (x[1]*tmp2 + x[2]*tmp3); tmp4 = tmp4*tmp4;
        jac_row[0] = -1;
        jac_row[1] = tmp1*tmp2/tmp4;
        jac_row[2] = tmp1*tmp3/tmp4;
        return 0;
    }
};

void testLmstr1()
{
  const int n=3;
  int info;

  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmstr_functor functor;
  LevenbergMarquardt<lmstr_functor> lm(functor);
  info = lm.lmstr1(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 6);
  VERIFY_IS_EQUAL(lm.njev, 5);

  // check norm
  VERIFY_IS_APPROX(lm.fvec.blueNorm(), 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695 ;
  VERIFY_IS_APPROX(x, x_ref);
}

void testLmstr()
{
  const int n=3;
  int info;
  double fnorm;
  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmstr_functor functor;
  LevenbergMarquardt<lmstr_functor> lm(functor);
  info = lm.minimizeOptimumStorage(x);

  // check return values
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 6);
  VERIFY_IS_EQUAL(lm.njev, 5);

  // check norm
  fnorm = lm.fvec.blueNorm();
  VERIFY_IS_APPROX(fnorm, 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  VERIFY_IS_APPROX(x, x_ref);

}

struct lmdif_functor : Functor<double>
{
    lmdif_functor(void) : Functor<double>(3,15) {}
    int operator()(const VectorXd &x, VectorXd &fvec) const
    {
        int i;
        double tmp1,tmp2,tmp3;
        static const double y[15]={1.4e-1,1.8e-1,2.2e-1,2.5e-1,2.9e-1,3.2e-1,3.5e-1,3.9e-1,
            3.7e-1,5.8e-1,7.3e-1,9.6e-1,1.34e0,2.1e0,4.39e0};

        assert(x.size()==3);
        assert(fvec.size()==15);
        for (i=0; i<15; i++)
        {
            tmp1 = i+1;
            tmp2 = 15 - i;
            tmp3 = tmp1;

            if (i >= 8) tmp3 = tmp2;
            fvec[i] = y[i] - (x[0] + tmp1/(x[1]*tmp2 + x[2]*tmp3));
        }
        return 0;
    }
};

void testLmdif1()
{
  const int n=3;
  int info;

  VectorXd x(n), fvec(15);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmdif_functor functor;
  DenseIndex nfev;
  info = LevenbergMarquardt<lmdif_functor>::lmdif1(functor, x, &nfev);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(nfev, 26);

  // check norm
  functor(x, fvec);
  VERIFY_IS_APPROX(fvec.blueNorm(), 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.0824106, 1.1330366, 2.3436947;
  VERIFY_IS_APPROX(x, x_ref);

}

void testLmdif()
{
  const int m=15, n=3;
  int info;
  double fnorm, covfac;
  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmdif_functor functor;
  NumericalDiff<lmdif_functor> numDiff(functor);
  LevenbergMarquardt<NumericalDiff<lmdif_functor> > lm(numDiff);
  info = lm.minimize(x);

  // check return values
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 26);

  // check norm
  fnorm = lm.fvec.blueNorm();
  VERIFY_IS_APPROX(fnorm, 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  VERIFY_IS_APPROX(x, x_ref);

  // check covariance
  covfac = fnorm*fnorm/(m-n);
  internal::covar(lm.fjac, lm.permutation.indices()); // TODO : move this as a function of lm

  MatrixXd cov_ref(n,n);
  cov_ref <<
      0.0001531202,   0.002869942,  -0.002656662,
      0.002869942,    0.09480937,   -0.09098997,
      -0.002656662,   -0.09098997,    0.08778729;

//  std::cout << fjac*covfac << std::endl;

  MatrixXd cov;
  cov =  covfac*lm.fjac.topLeftCorner<n,n>();
  VERIFY_IS_APPROX( cov, cov_ref);
  // TODO: why isn't this allowed ? :
  // VERIFY_IS_APPROX( covfac*fjac.topLeftCorner<n,n>() , cov_ref);
}

struct chwirut2_functor : Functor<double>
{
    chwirut2_functor(void) : Functor<double>(3,54) {}
    static const double m_x[54];
    static const double m_y[54];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        int i;

        assert(b.size()==3);
        assert(fvec.size()==54);
        for(i=0; i<54; i++) {
            double x = m_x[i];
            fvec[i] = exp(-b[0]*x)/(b[1]+b[2]*x) - m_y[i];
        }
        return 0;
    }
    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==3);
        assert(fjac.rows()==54);
        assert(fjac.cols()==3);
        for(int i=0; i<54; i++) {
            double x = m_x[i];
            double factor = 1./(b[1]+b[2]*x);
            double e = exp(-b[0]*x);
            fjac(i,0) = -x*e*factor;
            fjac(i,1) = -e*factor*factor;
            fjac(i,2) = -x*e*factor*factor;
        }
        return 0;
    }
};
const double chwirut2_functor::m_x[54] = { 0.500E0, 1.000E0, 1.750E0, 3.750E0, 5.750E0, 0.875E0, 2.250E0, 3.250E0, 5.250E0, 0.750E0, 1.750E0, 2.750E0, 4.750E0, 0.625E0, 1.250E0, 2.250E0, 4.250E0, .500E0, 3.000E0, .750E0, 3.000E0, 1.500E0, 6.000E0, 3.000E0, 6.000E0, 1.500E0, 3.000E0, .500E0, 2.000E0, 4.000E0, .750E0, 2.000E0, 5.000E0, .750E0, 2.250E0, 3.750E0, 5.750E0, 3.000E0, .750E0, 2.500E0, 4.000E0, .750E0, 2.500E0, 4.000E0, .750E0, 2.500E0, 4.000E0, .500E0, 6.000E0, 3.000E0, .500E0, 2.750E0, .500E0, 1.750E0};
const double chwirut2_functor::m_y[54] = { 92.9000E0 ,57.1000E0 ,31.0500E0 ,11.5875E0 ,8.0250E0 ,63.6000E0 ,21.4000E0 ,14.2500E0 ,8.4750E0 ,63.8000E0 ,26.8000E0 ,16.4625E0 ,7.1250E0 ,67.3000E0 ,41.0000E0 ,21.1500E0 ,8.1750E0 ,81.5000E0 ,13.1200E0 ,59.9000E0 ,14.6200E0 ,32.9000E0 ,5.4400E0 ,12.5600E0 ,5.4400E0 ,32.0000E0 ,13.9500E0 ,75.8000E0 ,20.0000E0 ,10.4200E0 ,59.5000E0 ,21.6700E0 ,8.5500E0 ,62.0000E0 ,20.2000E0 ,7.7600E0 ,3.7500E0 ,11.8100E0 ,54.7000E0 ,23.7000E0 ,11.5500E0 ,61.3000E0 ,17.7000E0 ,8.7400E0 ,59.2000E0 ,16.3000E0 ,8.6200E0 ,81.0000E0 ,4.8700E0 ,14.6200E0 ,81.7000E0 ,17.1700E0 ,81.3000E0 ,28.9000E0  };

// http://www.itl.nist.gov/div898/strd/nls/data/chwirut2.shtml
void testNistChwirut2(void)
{
  const int n=3;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x<< 0.1, 0.01, 0.02;
  // do the computation
  chwirut2_functor functor;
  LevenbergMarquardt<chwirut2_functor> lm(functor);
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 10);
  VERIFY_IS_EQUAL(lm.njev, 8);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.1304802941E+02);
  // check x
  VERIFY_IS_APPROX(x[0], 1.6657666537E-01);
  VERIFY_IS_APPROX(x[1], 5.1653291286E-03);
  VERIFY_IS_APPROX(x[2], 1.2150007096E-02);

  /*
   * Second try
   */
  x<< 0.15, 0.008, 0.010;
  // do the computation
  lm.resetParameters();
  lm.parameters.ftol = 1.E6*NumTraits<double>::epsilon();
  lm.parameters.xtol = 1.E6*NumTraits<double>::epsilon();
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 7);
  VERIFY_IS_EQUAL(lm.njev, 6);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.1304802941E+02);
  // check x
  VERIFY_IS_APPROX(x[0], 1.6657666537E-01);
  VERIFY_IS_APPROX(x[1], 5.1653291286E-03);
  VERIFY_IS_APPROX(x[2], 1.2150007096E-02);
}


struct misra1a_functor : Functor<double>
{
    misra1a_functor(void) : Functor<double>(2,14) {}
    static const double m_x[14];
    static const double m_y[14];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        assert(b.size()==2);
        assert(fvec.size()==14);
        for(int i=0; i<14; i++) {
            fvec[i] = b[0]*(1.-exp(-b[1]*m_x[i])) - m_y[i] ;
        }
        return 0;
    }
    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==2);
        assert(fjac.rows()==14);
        assert(fjac.cols()==2);
        for(int i=0; i<14; i++) {
            fjac(i,0) = (1.-exp(-b[1]*m_x[i]));
            fjac(i,1) = (b[0]*m_x[i]*exp(-b[1]*m_x[i]));
        }
        return 0;
    }
};
const double misra1a_functor::m_x[14] = { 77.6E0, 114.9E0, 141.1E0, 190.8E0, 239.9E0, 289.0E0, 332.8E0, 378.4E0, 434.8E0, 477.3E0, 536.8E0, 593.1E0, 689.1E0, 760.0E0};
const double misra1a_functor::m_y[14] = { 10.07E0, 14.73E0, 17.94E0, 23.93E0, 29.61E0, 35.18E0, 40.02E0, 44.82E0, 50.76E0, 55.05E0, 61.01E0, 66.40E0, 75.47E0, 81.78E0};

// http://www.itl.nist.gov/div898/strd/nls/data/misra1a.shtml
void testNistMisra1a(void)
{
  const int n=2;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x<< 500., 0.0001;
  // do the computation
  misra1a_functor functor;
  LevenbergMarquardt<misra1a_functor> lm(functor);
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 19);
  VERIFY_IS_EQUAL(lm.njev, 15);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.2455138894E-01);
  // check x
  VERIFY_IS_APPROX(x[0], 2.3894212918E+02);
  VERIFY_IS_APPROX(x[1], 5.5015643181E-04);

  /*
   * Second try
   */
  x<< 250., 0.0005;
  // do the computation
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 5);
  VERIFY_IS_EQUAL(lm.njev, 4);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.2455138894E-01);
  // check x
  VERIFY_IS_APPROX(x[0], 2.3894212918E+02);
  VERIFY_IS_APPROX(x[1], 5.5015643181E-04);
}

struct hahn1_functor : Functor<double>
{
    hahn1_functor(void) : Functor<double>(7,236) {}
    static const double m_x[236];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        static const double m_y[236] = { .591E0 , 1.547E0 , 2.902E0 , 2.894E0 , 4.703E0 , 6.307E0 , 7.03E0  , 7.898E0 , 9.470E0 , 9.484E0 , 10.072E0 , 10.163E0 , 11.615E0 , 12.005E0 , 12.478E0 , 12.982E0 , 12.970E0 , 13.926E0 , 14.452E0 , 14.404E0 , 15.190E0 , 15.550E0 , 15.528E0 , 15.499E0 , 16.131E0 , 16.438E0 , 16.387E0 , 16.549E0 , 16.872E0 , 16.830E0 , 16.926E0 , 16.907E0 , 16.966E0 , 17.060E0 , 17.122E0 , 17.311E0 , 17.355E0 , 17.668E0 , 17.767E0 , 17.803E0 , 17.765E0 , 17.768E0 , 17.736E0 , 17.858E0 , 17.877E0 , 17.912E0 , 18.046E0 , 18.085E0 , 18.291E0 , 18.357E0 , 18.426E0 , 18.584E0 , 18.610E0 , 18.870E0 , 18.795E0 , 19.111E0 , .367E0 , .796E0 , 0.892E0 , 1.903E0 , 2.150E0 , 3.697E0 , 5.870E0 , 6.421E0 , 7.422E0 , 9.944E0 , 11.023E0 , 11.87E0  , 12.786E0 , 14.067E0 , 13.974E0 , 14.462E0 , 14.464E0 , 15.381E0 , 15.483E0 , 15.59E0  , 16.075E0 , 16.347E0 , 16.181E0 , 16.915E0 , 17.003E0 , 16.978E0 , 17.756E0 , 17.808E0 , 17.868E0 , 18.481E0 , 18.486E0 , 19.090E0 , 16.062E0 , 16.337E0 , 16.345E0 ,
        16.388E0 , 17.159E0 , 17.116E0 , 17.164E0 , 17.123E0 , 17.979E0 , 17.974E0 , 18.007E0 , 17.993E0 , 18.523E0 , 18.669E0 , 18.617E0 , 19.371E0 , 19.330E0 , 0.080E0 , 0.248E0 , 1.089E0 , 1.418E0 , 2.278E0 , 3.624E0 , 4.574E0 , 5.556E0 , 7.267E0 , 7.695E0 , 9.136E0 , 9.959E0 , 9.957E0 , 11.600E0 , 13.138E0 , 13.564E0 , 13.871E0 , 13.994E0 , 14.947E0 , 15.473E0 , 15.379E0 , 15.455E0 , 15.908E0 , 16.114E0 , 17.071E0 , 17.135E0 , 17.282E0 , 17.368E0 , 17.483E0 , 17.764E0 , 18.185E0 , 18.271E0 , 18.236E0 , 18.237E0 , 18.523E0 , 18.627E0 , 18.665E0 , 19.086E0 , 0.214E0 , 0.943E0 , 1.429E0 , 2.241E0 , 2.951E0 , 3.782E0 , 4.757E0 , 5.602E0 , 7.169E0 , 8.920E0 , 10.055E0 , 12.035E0 , 12.861E0 , 13.436E0 , 14.167E0 , 14.755E0 , 15.168E0 , 15.651E0 , 15.746E0 , 16.216E0 , 16.445E0 , 16.965E0 , 17.121E0 , 17.206E0 , 17.250E0 , 17.339E0 , 17.793E0 , 18.123E0 , 18.49E0  , 18.566E0 , 18.645E0 , 18.706E0 , 18.924E0 , 19.1E0   , 0.375E0 , 0.471E0 , 1.504E0 , 2.204E0 , 2.813E0 , 4.765E0 , 9.835E0 , 10.040E0 , 11.946E0 , 12.596E0 , 
13.303E0 , 13.922E0 , 14.440E0 , 14.951E0 , 15.627E0 , 15.639E0 , 15.814E0 , 16.315E0 , 16.334E0 , 16.430E0 , 16.423E0 , 17.024E0 , 17.009E0 , 17.165E0 , 17.134E0 , 17.349E0 , 17.576E0 , 17.848E0 , 18.090E0 , 18.276E0 , 18.404E0 , 18.519E0 , 19.133E0 , 19.074E0 , 19.239E0 , 19.280E0 , 19.101E0 , 19.398E0 , 19.252E0 , 19.89E0  , 20.007E0 , 19.929E0 , 19.268E0 , 19.324E0 , 20.049E0 , 20.107E0 , 20.062E0 , 20.065E0 , 19.286E0 , 19.972E0 , 20.088E0 , 20.743E0 , 20.83E0  , 20.935E0 , 21.035E0 , 20.93E0  , 21.074E0 , 21.085E0 , 20.935E0 };

        //        int called=0; printf("call hahn1_functor with  iflag=%d, called=%d\n", iflag, called); if (iflag==1) called++;

        assert(b.size()==7);
        assert(fvec.size()==236);
        for(int i=0; i<236; i++) {
            double x=m_x[i], xx=x*x, xxx=xx*x;
            fvec[i] = (b[0]+b[1]*x+b[2]*xx+b[3]*xxx) / (1.+b[4]*x+b[5]*xx+b[6]*xxx) - m_y[i];
        }
        return 0;
    }

    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==7);
        assert(fjac.rows()==236);
        assert(fjac.cols()==7);
        for(int i=0; i<236; i++) {
            double x=m_x[i], xx=x*x, xxx=xx*x;
            double fact = 1./(1.+b[4]*x+b[5]*xx+b[6]*xxx);
            fjac(i,0) = 1.*fact;
            fjac(i,1) = x*fact;
            fjac(i,2) = xx*fact;
            fjac(i,3) = xxx*fact;
            fact = - (b[0]+b[1]*x+b[2]*xx+b[3]*xxx) * fact * fact;
            fjac(i,4) = x*fact;
            fjac(i,5) = xx*fact;
            fjac(i,6) = xxx*fact;
        }
        return 0;
    }
};
const double hahn1_functor::m_x[236] = { 24.41E0 , 34.82E0 , 44.09E0 , 45.07E0 , 54.98E0 , 65.51E0 , 70.53E0 , 75.70E0 , 89.57E0 , 91.14E0 , 96.40E0 , 97.19E0 , 114.26E0 , 120.25E0 , 127.08E0 , 133.55E0 , 133.61E0 , 158.67E0 , 172.74E0 , 171.31E0 , 202.14E0 , 220.55E0 , 221.05E0 , 221.39E0 , 250.99E0 , 268.99E0 , 271.80E0 , 271.97E0 , 321.31E0 , 321.69E0 , 330.14E0 , 333.03E0 , 333.47E0 , 340.77E0 , 345.65E0 , 373.11E0 , 373.79E0 , 411.82E0 , 419.51E0 , 421.59E0 , 422.02E0 , 422.47E0 , 422.61E0 , 441.75E0 , 447.41E0 , 448.7E0  , 472.89E0 , 476.69E0 , 522.47E0 , 522.62E0 , 524.43E0 , 546.75E0 , 549.53E0 , 575.29E0 , 576.00E0 , 625.55E0 , 20.15E0 , 28.78E0 , 29.57E0 , 37.41E0 , 39.12E0 , 50.24E0 , 61.38E0 , 66.25E0 , 73.42E0 , 95.52E0 , 107.32E0 , 122.04E0 , 134.03E0 , 163.19E0 , 163.48E0 , 175.70E0 , 179.86E0 , 211.27E0 , 217.78E0 , 219.14E0 , 262.52E0 , 268.01E0 , 268.62E0 , 336.25E0 , 337.23E0 , 339.33E0 , 427.38E0 , 428.58E0 , 432.68E0 , 528.99E0 , 531.08E0 , 628.34E0 , 253.24E0 , 273.13E0 , 273.66E0 ,
282.10E0 , 346.62E0 , 347.19E0 , 348.78E0 , 351.18E0 , 450.10E0 , 450.35E0 , 451.92E0 , 455.56E0 , 552.22E0 , 553.56E0 , 555.74E0 , 652.59E0 , 656.20E0 , 14.13E0 , 20.41E0 , 31.30E0 , 33.84E0 , 39.70E0 , 48.83E0 , 54.50E0 , 60.41E0 , 72.77E0 , 75.25E0 , 86.84E0 , 94.88E0 , 96.40E0 , 117.37E0 , 139.08E0 , 147.73E0 , 158.63E0 , 161.84E0 , 192.11E0 , 206.76E0 , 209.07E0 , 213.32E0 , 226.44E0 , 237.12E0 , 330.90E0 , 358.72E0 , 370.77E0 , 372.72E0 , 396.24E0 , 416.59E0 , 484.02E0 , 495.47E0 , 514.78E0 , 515.65E0 , 519.47E0 , 544.47E0 , 560.11E0 , 620.77E0 , 18.97E0 , 28.93E0 , 33.91E0 , 40.03E0 , 44.66E0 , 49.87E0 , 55.16E0 , 60.90E0 , 72.08E0 , 85.15E0 , 97.06E0 , 119.63E0 , 133.27E0 , 143.84E0 , 161.91E0 , 180.67E0 , 198.44E0 , 226.86E0 , 229.65E0 , 258.27E0 , 273.77E0 , 339.15E0 , 350.13E0 , 362.75E0 , 371.03E0 , 393.32E0 , 448.53E0 , 473.78E0 , 511.12E0 , 524.70E0 , 548.75E0 , 551.64E0 , 574.02E0 , 623.86E0 , 21.46E0 , 24.33E0 , 33.43E0 , 39.22E0 , 44.18E0 , 55.02E0 , 94.33E0 , 96.44E0 , 118.82E0 , 128.48E0 ,
141.94E0 , 156.92E0 , 171.65E0 , 190.00E0 , 223.26E0 , 223.88E0 , 231.50E0 , 265.05E0 , 269.44E0 , 271.78E0 , 273.46E0 , 334.61E0 , 339.79E0 , 349.52E0 , 358.18E0 , 377.98E0 , 394.77E0 , 429.66E0 , 468.22E0 , 487.27E0 , 519.54E0 , 523.03E0 , 612.99E0 , 638.59E0 , 641.36E0 , 622.05E0 , 631.50E0 , 663.97E0 , 646.9E0  , 748.29E0 , 749.21E0 , 750.14E0 , 647.04E0 , 646.89E0 , 746.9E0  , 748.43E0 , 747.35E0 , 749.27E0 , 647.61E0 , 747.78E0 , 750.51E0 , 851.37E0 , 845.97E0 , 847.54E0 , 849.93E0 , 851.61E0 , 849.75E0 , 850.98E0 , 848.23E0};

// http://www.itl.nist.gov/div898/strd/nls/data/hahn1.shtml
void testNistHahn1(void)
{
  const int  n=7;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x<< 10., -1., .05, -.00001, -.05, .001, -.000001;
  // do the computation
  hahn1_functor functor;
  LevenbergMarquardt<hahn1_functor> lm(functor);
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 11);
  VERIFY_IS_EQUAL(lm.njev, 10);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.5324382854E+00);
  // check x
  VERIFY_IS_APPROX(x[0], 1.0776351733E+00);
  VERIFY_IS_APPROX(x[1],-1.2269296921E-01);
  VERIFY_IS_APPROX(x[2], 4.0863750610E-03);
  VERIFY_IS_APPROX(x[3],-1.426264e-06); // shoulde be : -1.4262662514E-06
  VERIFY_IS_APPROX(x[4],-5.7609940901E-03);
  VERIFY_IS_APPROX(x[5], 2.4053735503E-04);
  VERIFY_IS_APPROX(x[6],-1.2314450199E-07);

  /*
   * Second try
   */
  x<< .1, -.1, .005, -.000001, -.005, .0001, -.0000001;
  // do the computation
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 11);
  VERIFY_IS_EQUAL(lm.njev, 10);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.5324382854E+00);
  // check x
  VERIFY_IS_APPROX(x[0], 1.077640); // should be :  1.0776351733E+00
  VERIFY_IS_APPROX(x[1], -0.1226933); // should be : -1.2269296921E-01
  VERIFY_IS_APPROX(x[2], 0.004086383); // should be : 4.0863750610E-03
  VERIFY_IS_APPROX(x[3], -1.426277e-06); // shoulde be : -1.4262662514E-06
  VERIFY_IS_APPROX(x[4],-5.7609940901E-03);
  VERIFY_IS_APPROX(x[5], 0.00024053772); // should be : 2.4053735503E-04
  VERIFY_IS_APPROX(x[6], -1.231450e-07); // should be : -1.2314450199E-07

}

struct misra1d_functor : Functor<double>
{
    misra1d_functor(void) : Functor<double>(2,14) {}
    static const double x[14];
    static const double y[14];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        assert(b.size()==2);
        assert(fvec.size()==14);
        for(int i=0; i<14; i++) {
            fvec[i] = b[0]*b[1]*x[i]/(1.+b[1]*x[i]) - y[i];
        }
        return 0;
    }
    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==2);
        assert(fjac.rows()==14);
        assert(fjac.cols()==2);
        for(int i=0; i<14; i++) {
            double den = 1.+b[1]*x[i];
            fjac(i,0) = b[1]*x[i] / den;
            fjac(i,1) = b[0]*x[i]*(den-b[1]*x[i])/den/den;
        }
        return 0;
    }
};
const double misra1d_functor::x[14] = { 77.6E0, 114.9E0, 141.1E0, 190.8E0, 239.9E0, 289.0E0, 332.8E0, 378.4E0, 434.8E0, 477.3E0, 536.8E0, 593.1E0, 689.1E0, 760.0E0};
const double misra1d_functor::y[14] = { 10.07E0, 14.73E0, 17.94E0, 23.93E0, 29.61E0, 35.18E0, 40.02E0, 44.82E0, 50.76E0, 55.05E0, 61.01E0, 66.40E0, 75.47E0, 81.78E0};

// http://www.itl.nist.gov/div898/strd/nls/data/misra1d.shtml
void testNistMisra1d(void)
{
  const int n=2;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x<< 500., 0.0001;
  // do the computation
  misra1d_functor functor;
  LevenbergMarquardt<misra1d_functor> lm(functor);
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 3);
  VERIFY_IS_EQUAL(lm.nfev, 9);
  VERIFY_IS_EQUAL(lm.njev, 7);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.6419295283E-02);
  // check x
  VERIFY_IS_APPROX(x[0], 4.3736970754E+02);
  VERIFY_IS_APPROX(x[1], 3.0227324449E-04);

  /*
   * Second try
   */
  x<< 450., 0.0003;
  // do the computation
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 4);
  VERIFY_IS_EQUAL(lm.njev, 3);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.6419295283E-02);
  // check x
  VERIFY_IS_APPROX(x[0], 4.3736970754E+02);
  VERIFY_IS_APPROX(x[1], 3.0227324449E-04);
}


struct lanczos1_functor : Functor<double>
{
    lanczos1_functor(void) : Functor<double>(6,24) {}
    static const double x[24];
    static const double y[24];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        assert(b.size()==6);
        assert(fvec.size()==24);
        for(int i=0; i<24; i++)
            fvec[i] = b[0]*exp(-b[1]*x[i]) + b[2]*exp(-b[3]*x[i]) + b[4]*exp(-b[5]*x[i])  - y[i];
        return 0;
    }
    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==6);
        assert(fjac.rows()==24);
        assert(fjac.cols()==6);
        for(int i=0; i<24; i++) {
            fjac(i,0) = exp(-b[1]*x[i]);
            fjac(i,1) = -b[0]*x[i]*exp(-b[1]*x[i]);
            fjac(i,2) = exp(-b[3]*x[i]);
            fjac(i,3) = -b[2]*x[i]*exp(-b[3]*x[i]);
            fjac(i,4) = exp(-b[5]*x[i]);
            fjac(i,5) = -b[4]*x[i]*exp(-b[5]*x[i]);
        }
        return 0;
    }
};
const double lanczos1_functor::x[24] = { 0.000000000000E+00, 5.000000000000E-02, 1.000000000000E-01, 1.500000000000E-01, 2.000000000000E-01, 2.500000000000E-01, 3.000000000000E-01, 3.500000000000E-01, 4.000000000000E-01, 4.500000000000E-01, 5.000000000000E-01, 5.500000000000E-01, 6.000000000000E-01, 6.500000000000E-01, 7.000000000000E-01, 7.500000000000E-01, 8.000000000000E-01, 8.500000000000E-01, 9.000000000000E-01, 9.500000000000E-01, 1.000000000000E+00, 1.050000000000E+00, 1.100000000000E+00, 1.150000000000E+00 };
const double lanczos1_functor::y[24] = { 2.513400000000E+00 ,2.044333373291E+00 ,1.668404436564E+00 ,1.366418021208E+00 ,1.123232487372E+00 ,9.268897180037E-01 ,7.679338563728E-01 ,6.388775523106E-01 ,5.337835317402E-01 ,4.479363617347E-01 ,3.775847884350E-01 ,3.197393199326E-01 ,2.720130773746E-01 ,2.324965529032E-01 ,1.996589546065E-01 ,1.722704126914E-01 ,1.493405660168E-01 ,1.300700206922E-01 ,1.138119324644E-01 ,1.000415587559E-01 ,8.833209084540E-02 ,7.833544019350E-02 ,6.976693743449E-02 ,6.239312536719E-02 };

// http://www.itl.nist.gov/div898/strd/nls/data/lanczos1.shtml
void testNistLanczos1(void)
{
  const int n=6;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x<< 1.2, 0.3, 5.6, 5.5, 6.5, 7.6;
  // do the computation
  lanczos1_functor functor;
  LevenbergMarquardt<lanczos1_functor> lm(functor);
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 2);
  VERIFY_IS_EQUAL(lm.nfev, 79);
  VERIFY_IS_EQUAL(lm.njev, 72);
  // check norm^2
  std::cout.precision(30);
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.4290986055242372e-25);  // should be 1.4307867721E-25, but nist results are on 128-bit floats
  // check x
  VERIFY_IS_APPROX(x[0], 9.5100000027E-02);
  VERIFY_IS_APPROX(x[1], 1.0000000001E+00);
  VERIFY_IS_APPROX(x[2], 8.6070000013E-01);
  VERIFY_IS_APPROX(x[3], 3.0000000002E+00);
  VERIFY_IS_APPROX(x[4], 1.5575999998E+00);
  VERIFY_IS_APPROX(x[5], 5.0000000001E+00);

  /*
   * Second try
   */
  x<< 0.5, 0.7, 3.6, 4.2, 4., 6.3;
  // do the computation
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 2);
  VERIFY_IS_EQUAL(lm.nfev, 9);
  VERIFY_IS_EQUAL(lm.njev, 8);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.430571737783119393e-25);  // should be 1.4307867721E-25, but nist results are on 128-bit floats
  // check x
  VERIFY_IS_APPROX(x[0], 9.5100000027E-02);
  VERIFY_IS_APPROX(x[1], 1.0000000001E+00);
  VERIFY_IS_APPROX(x[2], 8.6070000013E-01);
  VERIFY_IS_APPROX(x[3], 3.0000000002E+00);
  VERIFY_IS_APPROX(x[4], 1.5575999998E+00);
  VERIFY_IS_APPROX(x[5], 5.0000000001E+00);

}

struct rat42_functor : Functor<double>
{
    rat42_functor(void) : Functor<double>(3,9) {}
    static const double x[9];
    static const double y[9];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        assert(b.size()==3);
        assert(fvec.size()==9);
        for(int i=0; i<9; i++) {
            fvec[i] = b[0] / (1.+exp(b[1]-b[2]*x[i])) - y[i];
        }
        return 0;
    }

    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==3);
        assert(fjac.rows()==9);
        assert(fjac.cols()==3);
        for(int i=0; i<9; i++) {
            double e = exp(b[1]-b[2]*x[i]);
            fjac(i,0) = 1./(1.+e);
            fjac(i,1) = -b[0]*e/(1.+e)/(1.+e);
            fjac(i,2) = +b[0]*e*x[i]/(1.+e)/(1.+e);
        }
        return 0;
    }
};
const double rat42_functor::x[9] = { 9.000E0, 14.000E0, 21.000E0, 28.000E0, 42.000E0, 57.000E0, 63.000E0, 70.000E0, 79.000E0 };
const double rat42_functor::y[9] = { 8.930E0 ,10.800E0 ,18.590E0 ,22.330E0 ,39.350E0 ,56.110E0 ,61.730E0 ,64.620E0 ,67.080E0 };

// http://www.itl.nist.gov/div898/strd/nls/data/ratkowsky2.shtml
void testNistRat42(void)
{
  const int n=3;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x<< 100., 1., 0.1;
  // do the computation
  rat42_functor functor;
  LevenbergMarquardt<rat42_functor> lm(functor);
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 10);
  VERIFY_IS_EQUAL(lm.njev, 8);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 8.0565229338E+00);
  // check x
  VERIFY_IS_APPROX(x[0], 7.2462237576E+01);
  VERIFY_IS_APPROX(x[1], 2.6180768402E+00);
  VERIFY_IS_APPROX(x[2], 6.7359200066E-02);

  /*
   * Second try
   */
  x<< 75., 2.5, 0.07;
  // do the computation
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 6);
  VERIFY_IS_EQUAL(lm.njev, 5);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 8.0565229338E+00);
  // check x
  VERIFY_IS_APPROX(x[0], 7.2462237576E+01);
  VERIFY_IS_APPROX(x[1], 2.6180768402E+00);
  VERIFY_IS_APPROX(x[2], 6.7359200066E-02);
}

struct MGH10_functor : Functor<double>
{
    MGH10_functor(void) : Functor<double>(3,16) {}
    static const double x[16];
    static const double y[16];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        assert(b.size()==3);
        assert(fvec.size()==16);
        for(int i=0; i<16; i++)
            fvec[i] =  b[0] * exp(b[1]/(x[i]+b[2])) - y[i];
        return 0;
    }
    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==3);
        assert(fjac.rows()==16);
        assert(fjac.cols()==3);
        for(int i=0; i<16; i++) {
            double factor = 1./(x[i]+b[2]);
            double e = exp(b[1]*factor);
            fjac(i,0) = e;
            fjac(i,1) = b[0]*factor*e;
            fjac(i,2) = -b[1]*b[0]*factor*factor*e;
        }
        return 0;
    }
};
const double MGH10_functor::x[16] = { 5.000000E+01, 5.500000E+01, 6.000000E+01, 6.500000E+01, 7.000000E+01, 7.500000E+01, 8.000000E+01, 8.500000E+01, 9.000000E+01, 9.500000E+01, 1.000000E+02, 1.050000E+02, 1.100000E+02, 1.150000E+02, 1.200000E+02, 1.250000E+02 };
const double MGH10_functor::y[16] = { 3.478000E+04, 2.861000E+04, 2.365000E+04, 1.963000E+04, 1.637000E+04, 1.372000E+04, 1.154000E+04, 9.744000E+03, 8.261000E+03, 7.030000E+03, 6.005000E+03, 5.147000E+03, 4.427000E+03, 3.820000E+03, 3.307000E+03, 2.872000E+03 };

// http://www.itl.nist.gov/div898/strd/nls/data/mgh10.shtml
void testNistMGH10(void)
{
  const int n=3;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x<< 2., 400000., 25000.;
  // do the computation
  MGH10_functor functor;
  LevenbergMarquardt<MGH10_functor> lm(functor);
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 2); 
  VERIFY_IS_EQUAL(lm.nfev, 284 ); 
  VERIFY_IS_EQUAL(lm.njev, 249 ); 
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 8.7945855171E+01);
  // check x
  VERIFY_IS_APPROX(x[0], 5.6096364710E-03);
  VERIFY_IS_APPROX(x[1], 6.1813463463E+03);
  VERIFY_IS_APPROX(x[2], 3.4522363462E+02);

  /*
   * Second try
   */
  x<< 0.02, 4000., 250.;
  // do the computation
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 3);
  VERIFY_IS_EQUAL(lm.nfev, 126);
  VERIFY_IS_EQUAL(lm.njev, 116);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 8.7945855171E+01);
  // check x
  VERIFY_IS_APPROX(x[0], 5.6096364710E-03);
  VERIFY_IS_APPROX(x[1], 6.1813463463E+03);
  VERIFY_IS_APPROX(x[2], 3.4522363462E+02);
}


struct BoxBOD_functor : Functor<double>
{
    BoxBOD_functor(void) : Functor<double>(2,6) {}
    static const double x[6];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        static const double y[6] = { 109., 149., 149., 191., 213., 224. };
        assert(b.size()==2);
        assert(fvec.size()==6);
        for(int i=0; i<6; i++)
            fvec[i] =  b[0]*(1.-exp(-b[1]*x[i])) - y[i];
        return 0;
    }
    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==2);
        assert(fjac.rows()==6);
        assert(fjac.cols()==2);
        for(int i=0; i<6; i++) {
            double e = exp(-b[1]*x[i]);
            fjac(i,0) = 1.-e;
            fjac(i,1) = b[0]*x[i]*e;
        }
        return 0;
    }
};
const double BoxBOD_functor::x[6] = { 1., 2., 3., 5., 7., 10. };

// http://www.itl.nist.gov/div898/strd/nls/data/boxbod.shtml
void testNistBoxBOD(void)
{
  const int n=2;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x<< 1., 1.;
  // do the computation
  BoxBOD_functor functor;
  LevenbergMarquardt<BoxBOD_functor> lm(functor);
  lm.parameters.ftol = 1.E6*NumTraits<double>::epsilon();
  lm.parameters.xtol = 1.E6*NumTraits<double>::epsilon();
  lm.parameters.factor = 10.;
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY(lm.nfev < 31); // 31
  VERIFY(lm.njev < 25); // 25
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.1680088766E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 2.1380940889E+02);
  VERIFY_IS_APPROX(x[1], 5.4723748542E-01);

  /*
   * Second try
   */
  x<< 100., 0.75;
  // do the computation
  lm.resetParameters();
  lm.parameters.ftol = NumTraits<double>::epsilon();
  lm.parameters.xtol = NumTraits<double>::epsilon();
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1); 
  VERIFY_IS_EQUAL(lm.nfev, 15 ); 
  VERIFY_IS_EQUAL(lm.njev, 14 ); 
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.1680088766E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 2.1380940889E+02);
  VERIFY_IS_APPROX(x[1], 5.4723748542E-01);
}

struct MGH17_functor : Functor<double>
{
    MGH17_functor(void) : Functor<double>(5,33) {}
    static const double x[33];
    static const double y[33];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        assert(b.size()==5);
        assert(fvec.size()==33);
        for(int i=0; i<33; i++)
            fvec[i] =  b[0] + b[1]*exp(-b[3]*x[i]) +  b[2]*exp(-b[4]*x[i]) - y[i];
        return 0;
    }
    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==5);
        assert(fjac.rows()==33);
        assert(fjac.cols()==5);
        for(int i=0; i<33; i++) {
            fjac(i,0) = 1.;
            fjac(i,1) = exp(-b[3]*x[i]);
            fjac(i,2) = exp(-b[4]*x[i]);
            fjac(i,3) = -x[i]*b[1]*exp(-b[3]*x[i]);
            fjac(i,4) = -x[i]*b[2]*exp(-b[4]*x[i]);
        }
        return 0;
    }
};
const double MGH17_functor::x[33] = { 0.000000E+00, 1.000000E+01, 2.000000E+01, 3.000000E+01, 4.000000E+01, 5.000000E+01, 6.000000E+01, 7.000000E+01, 8.000000E+01, 9.000000E+01, 1.000000E+02, 1.100000E+02, 1.200000E+02, 1.300000E+02, 1.400000E+02, 1.500000E+02, 1.600000E+02, 1.700000E+02, 1.800000E+02, 1.900000E+02, 2.000000E+02, 2.100000E+02, 2.200000E+02, 2.300000E+02, 2.400000E+02, 2.500000E+02, 2.600000E+02, 2.700000E+02, 2.800000E+02, 2.900000E+02, 3.000000E+02, 3.100000E+02, 3.200000E+02 };
const double MGH17_functor::y[33] = { 8.440000E-01, 9.080000E-01, 9.320000E-01, 9.360000E-01, 9.250000E-01, 9.080000E-01, 8.810000E-01, 8.500000E-01, 8.180000E-01, 7.840000E-01, 7.510000E-01, 7.180000E-01, 6.850000E-01, 6.580000E-01, 6.280000E-01, 6.030000E-01, 5.800000E-01, 5.580000E-01, 5.380000E-01, 5.220000E-01, 5.060000E-01, 4.900000E-01, 4.780000E-01, 4.670000E-01, 4.570000E-01, 4.480000E-01, 4.380000E-01, 4.310000E-01, 4.240000E-01, 4.200000E-01, 4.140000E-01, 4.110000E-01, 4.060000E-01 };

// http://www.itl.nist.gov/div898/strd/nls/data/mgh17.shtml
void testNistMGH17(void)
{
  const int n=5;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x<< 50., 150., -100., 1., 2.;
  // do the computation
  MGH17_functor functor;
  LevenbergMarquardt<MGH17_functor> lm(functor);
  lm.parameters.ftol = NumTraits<double>::epsilon();
  lm.parameters.xtol = NumTraits<double>::epsilon();
  lm.parameters.maxfev = 1000;
  info = lm.minimize(x);

  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.4648946975E-05);
  // check x
  VERIFY_IS_APPROX(x[0], 3.7541005211E-01);
  VERIFY_IS_APPROX(x[1], 1.9358469127E+00);
  VERIFY_IS_APPROX(x[2], -1.4646871366E+00);
  VERIFY_IS_APPROX(x[3], 1.2867534640E-02);
  VERIFY_IS_APPROX(x[4], 2.2122699662E-02);
  
  // check return value
  VERIFY_IS_EQUAL(info, 2); 
  VERIFY(lm.nfev < 650);  // 602
  VERIFY(lm.njev < 600);  // 545

  /*
   * Second try
   */
  x<< 0.5  ,1.5  ,-1   ,0.01 ,0.02;
  // do the computation
  lm.resetParameters();
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 18);
  VERIFY_IS_EQUAL(lm.njev, 15);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.4648946975E-05);
  // check x
  VERIFY_IS_APPROX(x[0], 3.7541005211E-01);
  VERIFY_IS_APPROX(x[1], 1.9358469127E+00);
  VERIFY_IS_APPROX(x[2], -1.4646871366E+00);
  VERIFY_IS_APPROX(x[3], 1.2867534640E-02);
  VERIFY_IS_APPROX(x[4], 2.2122699662E-02);
}

struct MGH09_functor : Functor<double>
{
    MGH09_functor(void) : Functor<double>(4,11) {}
    static const double _x[11];
    static const double y[11];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        assert(b.size()==4);
        assert(fvec.size()==11);
        for(int i=0; i<11; i++) {
            double x = _x[i], xx=x*x;
            fvec[i] = b[0]*(xx+x*b[1])/(xx+x*b[2]+b[3]) - y[i];
        }
        return 0;
    }
    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==4);
        assert(fjac.rows()==11);
        assert(fjac.cols()==4);
        for(int i=0; i<11; i++) {
            double x = _x[i], xx=x*x;
            double factor = 1./(xx+x*b[2]+b[3]);
            fjac(i,0) = (xx+x*b[1]) * factor;
            fjac(i,1) = b[0]*x* factor;
            fjac(i,2) = - b[0]*(xx+x*b[1]) * x * factor * factor;
            fjac(i,3) = - b[0]*(xx+x*b[1]) * factor * factor;
        }
        return 0;
    }
};
const double MGH09_functor::_x[11] = { 4., 2., 1., 5.E-1 , 2.5E-01, 1.670000E-01, 1.250000E-01,  1.E-01, 8.330000E-02, 7.140000E-02, 6.250000E-02 };
const double MGH09_functor::y[11] = { 1.957000E-01, 1.947000E-01, 1.735000E-01, 1.600000E-01, 8.440000E-02, 6.270000E-02, 4.560000E-02, 3.420000E-02, 3.230000E-02, 2.350000E-02, 2.460000E-02 };

// http://www.itl.nist.gov/div898/strd/nls/data/mgh09.shtml
void testNistMGH09(void)
{
  const int n=4;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x<< 25., 39, 41.5, 39.;
  // do the computation
  MGH09_functor functor;
  LevenbergMarquardt<MGH09_functor> lm(functor);
  lm.parameters.maxfev = 1000;
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1); 
  VERIFY_IS_EQUAL(lm.nfev, 490 ); 
  VERIFY_IS_EQUAL(lm.njev, 376 ); 
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 3.0750560385E-04);
  // check x
  VERIFY_IS_APPROX(x[0], 0.1928077089); // should be 1.9280693458E-01
  VERIFY_IS_APPROX(x[1], 0.19126423573); // should be 1.9128232873E-01
  VERIFY_IS_APPROX(x[2], 0.12305309914); // should be 1.2305650693E-01
  VERIFY_IS_APPROX(x[3], 0.13605395375); // should be 1.3606233068E-01

  /*
   * Second try
   */
  x<< 0.25, 0.39, 0.415, 0.39;
  // do the computation
  lm.resetParameters();
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 18);
  VERIFY_IS_EQUAL(lm.njev, 16);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 3.0750560385E-04);
  // check x
  VERIFY_IS_APPROX(x[0], 0.19280781); // should be 1.9280693458E-01
  VERIFY_IS_APPROX(x[1], 0.19126265); // should be 1.9128232873E-01
  VERIFY_IS_APPROX(x[2], 0.12305280); // should be 1.2305650693E-01
  VERIFY_IS_APPROX(x[3], 0.13605322); // should be 1.3606233068E-01
}



struct Bennett5_functor : Functor<double>
{
    Bennett5_functor(void) : Functor<double>(3,154) {}
    static const double x[154];
    static const double y[154];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        assert(b.size()==3);
        assert(fvec.size()==154);
        for(int i=0; i<154; i++)
            fvec[i] = b[0]* pow(b[1]+x[i],-1./b[2]) - y[i];
        return 0;
    }
    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==3);
        assert(fjac.rows()==154);
        assert(fjac.cols()==3);
        for(int i=0; i<154; i++) {
            double e = pow(b[1]+x[i],-1./b[2]);
            fjac(i,0) = e;
            fjac(i,1) = - b[0]*e/b[2]/(b[1]+x[i]);
            fjac(i,2) = b[0]*e*log(b[1]+x[i])/b[2]/b[2];
        }
        return 0;
    }
};
const double Bennett5_functor::x[154] = { 7.447168E0, 8.102586E0, 8.452547E0, 8.711278E0, 8.916774E0, 9.087155E0, 9.232590E0, 9.359535E0, 9.472166E0, 9.573384E0, 9.665293E0, 9.749461E0, 9.827092E0, 9.899128E0, 9.966321E0, 10.029280E0, 10.088510E0, 10.144430E0, 10.197380E0, 10.247670E0, 10.295560E0, 10.341250E0, 10.384950E0, 10.426820E0, 10.467000E0, 10.505640E0, 10.542830E0, 10.578690E0, 10.613310E0, 10.646780E0, 10.679150E0, 10.710520E0, 10.740920E0, 10.770440E0, 10.799100E0, 10.826970E0, 10.854080E0, 10.880470E0, 10.906190E0, 10.931260E0, 10.955720E0, 10.979590E0, 11.002910E0, 11.025700E0, 11.047980E0, 11.069770E0, 11.091100E0, 11.111980E0, 11.132440E0, 11.152480E0, 11.172130E0, 11.191410E0, 11.210310E0, 11.228870E0, 11.247090E0, 11.264980E0, 11.282560E0, 11.299840E0, 11.316820E0, 11.333520E0, 11.349940E0, 11.366100E0, 11.382000E0, 11.397660E0, 11.413070E0, 11.428240E0, 11.443200E0, 11.457930E0, 11.472440E0, 11.486750E0, 11.500860E0, 11.514770E0, 11.528490E0, 11.542020E0, 11.555380E0, 11.568550E0,
11.581560E0, 11.594420E0, 11.607121E0, 11.619640E0, 11.632000E0, 11.644210E0, 11.656280E0, 11.668200E0, 11.679980E0, 11.691620E0, 11.703130E0, 11.714510E0, 11.725760E0, 11.736880E0, 11.747890E0, 11.758780E0, 11.769550E0, 11.780200E0, 11.790730E0, 11.801160E0, 11.811480E0, 11.821700E0, 11.831810E0, 11.841820E0, 11.851730E0, 11.861550E0, 11.871270E0, 11.880890E0, 11.890420E0, 11.899870E0, 11.909220E0, 11.918490E0, 11.927680E0, 11.936780E0, 11.945790E0, 11.954730E0, 11.963590E0, 11.972370E0, 11.981070E0, 11.989700E0, 11.998260E0, 12.006740E0, 12.015150E0, 12.023490E0, 12.031760E0, 12.039970E0, 12.048100E0, 12.056170E0, 12.064180E0, 12.072120E0, 12.080010E0, 12.087820E0, 12.095580E0, 12.103280E0, 12.110920E0, 12.118500E0, 12.126030E0, 12.133500E0, 12.140910E0, 12.148270E0, 12.155570E0, 12.162830E0, 12.170030E0, 12.177170E0, 12.184270E0, 12.191320E0, 12.198320E0, 12.205270E0, 12.212170E0, 12.219030E0, 12.225840E0, 12.232600E0, 12.239320E0, 12.245990E0, 12.252620E0, 12.259200E0, 12.265750E0, 12.272240E0 };
const double Bennett5_functor::y[154] = { -34.834702E0 ,-34.393200E0 ,-34.152901E0 ,-33.979099E0 ,-33.845901E0 ,-33.732899E0 ,-33.640301E0 ,-33.559200E0 ,-33.486801E0 ,-33.423100E0 ,-33.365101E0 ,-33.313000E0 ,-33.260899E0 ,-33.217400E0 ,-33.176899E0 ,-33.139198E0 ,-33.101601E0 ,-33.066799E0 ,-33.035000E0 ,-33.003101E0 ,-32.971298E0 ,-32.942299E0 ,-32.916302E0 ,-32.890202E0 ,-32.864101E0 ,-32.841000E0 ,-32.817799E0 ,-32.797501E0 ,-32.774300E0 ,-32.757000E0 ,-32.733799E0 ,-32.716400E0 ,-32.699100E0 ,-32.678799E0 ,-32.661400E0 ,-32.644001E0 ,-32.626701E0 ,-32.612202E0 ,-32.597698E0 ,-32.583199E0 ,-32.568699E0 ,-32.554298E0 ,-32.539799E0 ,-32.525299E0 ,-32.510799E0 ,-32.499199E0 ,-32.487598E0 ,-32.473202E0 ,-32.461601E0 ,-32.435501E0 ,-32.435501E0 ,-32.426800E0 ,-32.412300E0 ,-32.400799E0 ,-32.392101E0 ,-32.380501E0 ,-32.366001E0 ,-32.357300E0 ,-32.348598E0 ,-32.339901E0 ,-32.328400E0 ,-32.319698E0 ,-32.311001E0 ,-32.299400E0 ,-32.290699E0 ,-32.282001E0 ,-32.273300E0 ,-32.264599E0 ,-32.256001E0 ,-32.247299E0
,-32.238602E0 ,-32.229900E0 ,-32.224098E0 ,-32.215401E0 ,-32.203800E0 ,-32.198002E0 ,-32.189400E0 ,-32.183601E0 ,-32.174900E0 ,-32.169102E0 ,-32.163300E0 ,-32.154598E0 ,-32.145901E0 ,-32.140099E0 ,-32.131401E0 ,-32.125599E0 ,-32.119801E0 ,-32.111198E0 ,-32.105400E0 ,-32.096699E0 ,-32.090900E0 ,-32.088001E0 ,-32.079300E0 ,-32.073502E0 ,-32.067699E0 ,-32.061901E0 ,-32.056099E0 ,-32.050301E0 ,-32.044498E0 ,-32.038799E0 ,-32.033001E0 ,-32.027199E0 ,-32.024300E0 ,-32.018501E0 ,-32.012699E0 ,-32.004002E0 ,-32.001099E0 ,-31.995300E0 ,-31.989500E0 ,-31.983700E0 ,-31.977900E0 ,-31.972099E0 ,-31.969299E0 ,-31.963501E0 ,-31.957701E0 ,-31.951900E0 ,-31.946100E0 ,-31.940300E0 ,-31.937401E0 ,-31.931601E0 ,-31.925800E0 ,-31.922899E0 ,-31.917101E0 ,-31.911301E0 ,-31.908400E0 ,-31.902599E0 ,-31.896900E0 ,-31.893999E0 ,-31.888201E0 ,-31.885300E0 ,-31.882401E0 ,-31.876600E0 ,-31.873699E0 ,-31.867901E0 ,-31.862101E0 ,-31.859200E0 ,-31.856300E0 ,-31.850500E0 ,-31.844700E0 ,-31.841801E0 ,-31.838900E0 ,-31.833099E0 ,-31.830200E0 ,
-31.827299E0 ,-31.821600E0 ,-31.818701E0 ,-31.812901E0 ,-31.809999E0 ,-31.807100E0 ,-31.801300E0 ,-31.798401E0 ,-31.795500E0 ,-31.789700E0 ,-31.786800E0 };

// http://www.itl.nist.gov/div898/strd/nls/data/bennett5.shtml
void testNistBennett5(void)
{
  const int  n=3;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x<< -2000., 50., 0.8;
  // do the computation
  Bennett5_functor functor;
  LevenbergMarquardt<Bennett5_functor> lm(functor);
  lm.parameters.maxfev = 1000;
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 758);
  VERIFY_IS_EQUAL(lm.njev, 744);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.2404744073E-04);
  // check x
  VERIFY_IS_APPROX(x[0], -2.5235058043E+03);
  VERIFY_IS_APPROX(x[1], 4.6736564644E+01);
  VERIFY_IS_APPROX(x[2], 9.3218483193E-01);
  /*
   * Second try
   */
  x<< -1500., 45., 0.85;
  // do the computation
  lm.resetParameters();
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 203);
  VERIFY_IS_EQUAL(lm.njev, 192);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.2404744073E-04);
  // check x
  VERIFY_IS_APPROX(x[0], -2523.3007865); // should be -2.5235058043E+03
  VERIFY_IS_APPROX(x[1], 46.735705771); // should be 4.6736564644E+01);
  VERIFY_IS_APPROX(x[2], 0.93219881891); // should be 9.3218483193E-01);
}

struct thurber_functor : Functor<double>
{
    thurber_functor(void) : Functor<double>(7,37) {}
    static const double _x[37];
    static const double _y[37];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        //        int called=0; printf("call hahn1_functor with  iflag=%d, called=%d\n", iflag, called); if (iflag==1) called++;
        assert(b.size()==7);
        assert(fvec.size()==37);
        for(int i=0; i<37; i++) {
            double x=_x[i], xx=x*x, xxx=xx*x;
            fvec[i] = (b[0]+b[1]*x+b[2]*xx+b[3]*xxx) / (1.+b[4]*x+b[5]*xx+b[6]*xxx) - _y[i];
        }
        return 0;
    }
    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==7);
        assert(fjac.rows()==37);
        assert(fjac.cols()==7);
        for(int i=0; i<37; i++) {
            double x=_x[i], xx=x*x, xxx=xx*x;
            double fact = 1./(1.+b[4]*x+b[5]*xx+b[6]*xxx);
            fjac(i,0) = 1.*fact;
            fjac(i,1) = x*fact;
            fjac(i,2) = xx*fact;
            fjac(i,3) = xxx*fact;
            fact = - (b[0]+b[1]*x+b[2]*xx+b[3]*xxx) * fact * fact;
            fjac(i,4) = x*fact;
            fjac(i,5) = xx*fact;
            fjac(i,6) = xxx*fact;
        }
        return 0;
    }
};
const double thurber_functor::_x[37] = { -3.067E0, -2.981E0, -2.921E0, -2.912E0, -2.840E0, -2.797E0, -2.702E0, -2.699E0, -2.633E0, -2.481E0, -2.363E0, -2.322E0, -1.501E0, -1.460E0, -1.274E0, -1.212E0, -1.100E0, -1.046E0, -0.915E0, -0.714E0, -0.566E0, -0.545E0, -0.400E0, -0.309E0, -0.109E0, -0.103E0, 0.010E0, 0.119E0, 0.377E0, 0.790E0, 0.963E0, 1.006E0, 1.115E0, 1.572E0, 1.841E0, 2.047E0, 2.200E0 };
const double thurber_functor::_y[37] = { 80.574E0, 84.248E0, 87.264E0, 87.195E0, 89.076E0, 89.608E0, 89.868E0, 90.101E0, 92.405E0, 95.854E0, 100.696E0, 101.060E0, 401.672E0, 390.724E0, 567.534E0, 635.316E0, 733.054E0, 759.087E0, 894.206E0, 990.785E0, 1090.109E0, 1080.914E0, 1122.643E0, 1178.351E0, 1260.531E0, 1273.514E0, 1288.339E0, 1327.543E0, 1353.863E0, 1414.509E0, 1425.208E0, 1421.384E0, 1442.962E0, 1464.350E0, 1468.705E0, 1447.894E0, 1457.628E0};

// http://www.itl.nist.gov/div898/strd/nls/data/thurber.shtml
void testNistThurber(void)
{
  const int n=7;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x<< 1000 ,1000 ,400 ,40 ,0.7,0.3,0.0 ;
  // do the computation
  thurber_functor functor;
  LevenbergMarquardt<thurber_functor> lm(functor);
  lm.parameters.ftol = 1.E4*NumTraits<double>::epsilon();
  lm.parameters.xtol = 1.E4*NumTraits<double>::epsilon();
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 39);
  VERIFY_IS_EQUAL(lm.njev, 36);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.6427082397E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 1.2881396800E+03);
  VERIFY_IS_APPROX(x[1], 1.4910792535E+03);
  VERIFY_IS_APPROX(x[2], 5.8323836877E+02);
  VERIFY_IS_APPROX(x[3], 7.5416644291E+01);
  VERIFY_IS_APPROX(x[4], 9.6629502864E-01);
  VERIFY_IS_APPROX(x[5], 3.9797285797E-01);
  VERIFY_IS_APPROX(x[6], 4.9727297349E-02);

  /*
   * Second try
   */
  x<< 1300 ,1500 ,500  ,75   ,1    ,0.4  ,0.05  ;
  // do the computation
  lm.resetParameters();
  lm.parameters.ftol = 1.E4*NumTraits<double>::epsilon();
  lm.parameters.xtol = 1.E4*NumTraits<double>::epsilon();
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 29);
  VERIFY_IS_EQUAL(lm.njev, 28);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.6427082397E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 1.2881396800E+03);
  VERIFY_IS_APPROX(x[1], 1.4910792535E+03);
  VERIFY_IS_APPROX(x[2], 5.8323836877E+02);
  VERIFY_IS_APPROX(x[3], 7.5416644291E+01);
  VERIFY_IS_APPROX(x[4], 9.6629502864E-01);
  VERIFY_IS_APPROX(x[5], 3.9797285797E-01);
  VERIFY_IS_APPROX(x[6], 4.9727297349E-02);
}

struct rat43_functor : Functor<double>
{
    rat43_functor(void) : Functor<double>(4,15) {}
    static const double x[15];
    static const double y[15];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        assert(b.size()==4);
        assert(fvec.size()==15);
        for(int i=0; i<15; i++)
            fvec[i] = b[0] * pow(1.+exp(b[1]-b[2]*x[i]),-1./b[3]) - y[i];
        return 0;
    }
    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==4);
        assert(fjac.rows()==15);
        assert(fjac.cols()==4);
        for(int i=0; i<15; i++) {
            double e = exp(b[1]-b[2]*x[i]);
            double power = -1./b[3];
            fjac(i,0) = pow(1.+e, power);
            fjac(i,1) = power*b[0]*e*pow(1.+e, power-1.);
            fjac(i,2) = -power*b[0]*e*x[i]*pow(1.+e, power-1.);
            fjac(i,3) = b[0]*power*power*log(1.+e)*pow(1.+e, power);
        }
        return 0;
    }
};
const double rat43_functor::x[15] = { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15. };
const double rat43_functor::y[15] = { 16.08, 33.83, 65.80, 97.20, 191.55, 326.20, 386.87, 520.53, 590.03, 651.92, 724.93, 699.56, 689.96, 637.56, 717.41 };

// http://www.itl.nist.gov/div898/strd/nls/data/ratkowsky3.shtml
void testNistRat43(void)
{
  const int n=4;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x<< 100., 10., 1., 1.;
  // do the computation
  rat43_functor functor;
  LevenbergMarquardt<rat43_functor> lm(functor);
  lm.parameters.ftol = 1.E6*NumTraits<double>::epsilon();
  lm.parameters.xtol = 1.E6*NumTraits<double>::epsilon();
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 27);
  VERIFY_IS_EQUAL(lm.njev, 20);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 8.7864049080E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 6.9964151270E+02);
  VERIFY_IS_APPROX(x[1], 5.2771253025E+00);
  VERIFY_IS_APPROX(x[2], 7.5962938329E-01);
  VERIFY_IS_APPROX(x[3], 1.2792483859E+00);

  /*
   * Second try
   */
  x<< 700., 5., 0.75, 1.3;
  // do the computation
  lm.resetParameters();
  lm.parameters.ftol = 1.E5*NumTraits<double>::epsilon();
  lm.parameters.xtol = 1.E5*NumTraits<double>::epsilon();
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 9);
  VERIFY_IS_EQUAL(lm.njev, 8);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 8.7864049080E+03);
  // check x
  VERIFY_IS_APPROX(x[0], 6.9964151270E+02);
  VERIFY_IS_APPROX(x[1], 5.2771253025E+00);
  VERIFY_IS_APPROX(x[2], 7.5962938329E-01);
  VERIFY_IS_APPROX(x[3], 1.2792483859E+00);
}



struct eckerle4_functor : Functor<double>
{
    eckerle4_functor(void) : Functor<double>(3,35) {}
    static const double x[35];
    static const double y[35];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        assert(b.size()==3);
        assert(fvec.size()==35);
        for(int i=0; i<35; i++)
            fvec[i] = b[0]/b[1] * exp(-0.5*(x[i]-b[2])*(x[i]-b[2])/(b[1]*b[1])) - y[i];
        return 0;
    }
    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==3);
        assert(fjac.rows()==35);
        assert(fjac.cols()==3);
        for(int i=0; i<35; i++) {
            double b12 = b[1]*b[1];
            double e = exp(-0.5*(x[i]-b[2])*(x[i]-b[2])/b12);
            fjac(i,0) = e / b[1];
            fjac(i,1) = ((x[i]-b[2])*(x[i]-b[2])/b12-1.) * b[0]*e/b12;
            fjac(i,2) = (x[i]-b[2])*e*b[0]/b[1]/b12;
        }
        return 0;
    }
};
const double eckerle4_functor::x[35] = { 400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0, 436.5, 438.0, 439.5, 441.0, 442.5, 444.0, 445.5, 447.0, 448.5, 450.0, 451.5, 453.0, 454.5, 456.0, 457.5, 459.0, 460.5, 462.0, 463.5, 465.0, 470.0, 475.0, 480.0, 485.0, 490.0, 495.0, 500.0};
const double eckerle4_functor::y[35] = { 0.0001575, 0.0001699, 0.0002350, 0.0003102, 0.0004917, 0.0008710, 0.0017418, 0.0046400, 0.0065895, 0.0097302, 0.0149002, 0.0237310, 0.0401683, 0.0712559, 0.1264458, 0.2073413, 0.2902366, 0.3445623, 0.3698049, 0.3668534, 0.3106727, 0.2078154, 0.1164354, 0.0616764, 0.0337200, 0.0194023, 0.0117831, 0.0074357, 0.0022732, 0.0008800, 0.0004579, 0.0002345, 0.0001586, 0.0001143, 0.0000710 };

// http://www.itl.nist.gov/div898/strd/nls/data/eckerle4.shtml
void testNistEckerle4(void)
{
  const int n=3;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x<< 1., 10., 500.;
  // do the computation
  eckerle4_functor functor;
  LevenbergMarquardt<eckerle4_functor> lm(functor);
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 18);
  VERIFY_IS_EQUAL(lm.njev, 15);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.4635887487E-03);
  // check x
  VERIFY_IS_APPROX(x[0], 1.5543827178);
  VERIFY_IS_APPROX(x[1], 4.0888321754);
  VERIFY_IS_APPROX(x[2], 4.5154121844E+02);

  /*
   * Second try
   */
  x<< 1.5, 5., 450.;
  // do the computation
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 7);
  VERIFY_IS_EQUAL(lm.njev, 6);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 1.4635887487E-03);
  // check x
  VERIFY_IS_APPROX(x[0], 1.5543827178);
  VERIFY_IS_APPROX(x[1], 4.0888321754);
  VERIFY_IS_APPROX(x[2], 4.5154121844E+02);
}

void test_NonLinearOptimization()
{
    // Tests using the examples provided by (c)minpack
    CALL_SUBTEST/*_1*/(testChkder());
    CALL_SUBTEST/*_1*/(testLmder1());
    CALL_SUBTEST/*_1*/(testLmder());
    CALL_SUBTEST/*_2*/(testHybrj1());
    CALL_SUBTEST/*_2*/(testHybrj());
    CALL_SUBTEST/*_2*/(testHybrd1());
    CALL_SUBTEST/*_2*/(testHybrd());
    CALL_SUBTEST/*_3*/(testLmstr1());
    CALL_SUBTEST/*_3*/(testLmstr());
    CALL_SUBTEST/*_3*/(testLmdif1());
    CALL_SUBTEST/*_3*/(testLmdif());

    // NIST tests, level of difficulty = "Lower"
    CALL_SUBTEST/*_4*/(testNistMisra1a());
    CALL_SUBTEST/*_4*/(testNistChwirut2());

    // NIST tests, level of difficulty = "Average"
    CALL_SUBTEST/*_5*/(testNistHahn1());
    CALL_SUBTEST/*_6*/(testNistMisra1d());
    CALL_SUBTEST/*_7*/(testNistMGH17());
    CALL_SUBTEST/*_8*/(testNistLanczos1());

//     // NIST tests, level of difficulty = "Higher"
    CALL_SUBTEST/*_9*/(testNistRat42());
//     CALL_SUBTEST/*_10*/(testNistMGH10());
    CALL_SUBTEST/*_11*/(testNistBoxBOD());
//     CALL_SUBTEST/*_12*/(testNistMGH09());
    CALL_SUBTEST/*_13*/(testNistBennett5());
    CALL_SUBTEST/*_14*/(testNistThurber());
    CALL_SUBTEST/*_15*/(testNistRat43());
    CALL_SUBTEST/*_16*/(testNistEckerle4());
}

/*
 * Can be useful for debugging...
  printf("info, nfev : %d, %d\n", info, lm.nfev);
  printf("info, nfev, njev : %d, %d, %d\n", info, solver.nfev, solver.njev);
  printf("info, nfev : %d, %d\n", info, solver.nfev);
  printf("x[0] : %.32g\n", x[0]);
  printf("x[1] : %.32g\n", x[1]);
  printf("x[2] : %.32g\n", x[2]);
  printf("x[3] : %.32g\n", x[3]);
  printf("fvec.blueNorm() : %.32g\n", solver.fvec.blueNorm());
  printf("fvec.blueNorm() : %.32g\n", lm.fvec.blueNorm());

  printf("info, nfev, njev : %d, %d, %d\n", info, lm.nfev, lm.njev);
  printf("fvec.squaredNorm() : %.13g\n", lm.fvec.squaredNorm());
  std::cout << x << std::endl;
  std::cout.precision(9);
  std::cout << x[0] << std::endl;
  std::cout << x[1] << std::endl;
  std::cout << x[2] << std::endl;
  std::cout << x[3] << std::endl;
*/

