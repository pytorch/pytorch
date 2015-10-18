// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>

#include <stdio.h>

#include "main.h"
#include <unsupported/Eigen/NumericalDiff>
    
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
  
  int m_inputs, m_values;
  
  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}
  
  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

};

struct my_functor : Functor<double>
{
    my_functor(void): Functor<double>(3,15) {}
    int operator()(const VectorXd &x, VectorXd &fvec) const
    {
        double tmp1, tmp2, tmp3;
        double y[15] = {1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
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

    int actual_df(const VectorXd &x, MatrixXd &fjac) const
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

void test_forward()
{
    VectorXd x(3);
    MatrixXd jac(15,3);
    MatrixXd actual_jac(15,3);
    my_functor functor;

    x << 0.082, 1.13, 2.35;

    // real one 
    functor.actual_df(x, actual_jac);
//    std::cout << actual_jac << std::endl << std::endl;

    // using NumericalDiff
    NumericalDiff<my_functor> numDiff(functor);
    numDiff.df(x, jac);
//    std::cout << jac << std::endl;

    VERIFY_IS_APPROX(jac, actual_jac);
}

void test_central()
{
    VectorXd x(3);
    MatrixXd jac(15,3);
    MatrixXd actual_jac(15,3);
    my_functor functor;

    x << 0.082, 1.13, 2.35;

    // real one 
    functor.actual_df(x, actual_jac);

    // using NumericalDiff
    NumericalDiff<my_functor,Central> numDiff(functor);
    numDiff.df(x, jac);

    VERIFY_IS_APPROX(jac, actual_jac);
}

void test_NumericalDiff()
{
    CALL_SUBTEST(test_forward());
    CALL_SUBTEST(test_central());
}
