// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <unsupported/Eigen/AutoDiff>

template<typename Scalar>
EIGEN_DONT_INLINE Scalar foo(const Scalar& x, const Scalar& y)
{
  using namespace std;
//   return x+std::sin(y);
  EIGEN_ASM_COMMENT("mybegin");
  return static_cast<Scalar>(x*2 - pow(x,2) + 2*sqrt(y*y) - 4 * sin(x) + 2 * cos(y) - exp(-0.5*x*x));
  //return x+2*y*x;//x*2 -std::pow(x,2);//(2*y/x);// - y*2;
  EIGEN_ASM_COMMENT("myend");
}

template<typename Vector>
EIGEN_DONT_INLINE typename Vector::Scalar foo(const Vector& p)
{
  typedef typename Vector::Scalar Scalar;
  return (p-Vector(Scalar(-1),Scalar(1.))).norm() + (p.array() * p.array()).sum() + p.dot(p);
}

template<typename _Scalar, int NX=Dynamic, int NY=Dynamic>
struct TestFunc1
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

  TestFunc1() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  TestFunc1(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

  template<typename T>
  void operator() (const Matrix<T,InputsAtCompileTime,1>& x, Matrix<T,ValuesAtCompileTime,1>* _v) const
  {
    Matrix<T,ValuesAtCompileTime,1>& v = *_v;

    v[0] = 2 * x[0] * x[0] + x[0] * x[1];
    v[1] = 3 * x[1] * x[0] + 0.5 * x[1] * x[1];
    if(inputs()>2)
    {
      v[0] += 0.5 * x[2];
      v[1] += x[2];
    }
    if(values()>2)
    {
      v[2] = 3 * x[1] * x[0] * x[0];
    }
    if (inputs()>2 && values()>2)
      v[2] *= x[2];
  }

  void operator() (const InputType& x, ValueType* v, JacobianType* _j) const
  {
    (*this)(x, v);

    if(_j)
    {
      JacobianType& j = *_j;

      j(0,0) = 4 * x[0] + x[1];
      j(1,0) = 3 * x[1];

      j(0,1) = x[0];
      j(1,1) = 3 * x[0] + 2 * 0.5 * x[1];

      if (inputs()>2)
      {
        j(0,2) = 0.5;
        j(1,2) = 1;
      }
      if(values()>2)
      {
        j(2,0) = 3 * x[1] * 2 * x[0];
        j(2,1) = 3 * x[0] * x[0];
      }
      if (inputs()>2 && values()>2)
      {
        j(2,0) *= x[2];
        j(2,1) *= x[2];

        j(2,2) = 3 * x[1] * x[0] * x[0];
        j(2,2) = 3 * x[1] * x[0] * x[0];
      }
    }
  }
};

template<typename Func> void forward_jacobian(const Func& f)
{
    typename Func::InputType x = Func::InputType::Random(f.inputs());
    typename Func::ValueType y(f.values()), yref(f.values());
    typename Func::JacobianType j(f.values(),f.inputs()), jref(f.values(),f.inputs());

    jref.setZero();
    yref.setZero();
    f(x,&yref,&jref);
//     std::cerr << y.transpose() << "\n\n";;
//     std::cerr << j << "\n\n";;

    j.setZero();
    y.setZero();
    AutoDiffJacobian<Func> autoj(f);
    autoj(x, &y, &j);
//     std::cerr << y.transpose() << "\n\n";;
//     std::cerr << j << "\n\n";;

    VERIFY_IS_APPROX(y, yref);
    VERIFY_IS_APPROX(j, jref);
}


// TODO also check actual derivatives!
void test_autodiff_scalar()
{
  Vector2f p = Vector2f::Random();
  typedef AutoDiffScalar<Vector2f> AD;
  AD ax(p.x(),Vector2f::UnitX());
  AD ay(p.y(),Vector2f::UnitY());
  AD res = foo<AD>(ax,ay);
  VERIFY_IS_APPROX(res.value(), foo(p.x(),p.y()));
}

// TODO also check actual derivatives!
void test_autodiff_vector()
{
  Vector2f p = Vector2f::Random();
  typedef AutoDiffScalar<Vector2f> AD;
  typedef Matrix<AD,2,1> VectorAD;
  VectorAD ap = p.cast<AD>();
  ap.x().derivatives() = Vector2f::UnitX();
  ap.y().derivatives() = Vector2f::UnitY();
  
  AD res = foo<VectorAD>(ap);
  VERIFY_IS_APPROX(res.value(), foo(p));
}

void test_autodiff_jacobian()
{
  CALL_SUBTEST(( forward_jacobian(TestFunc1<double,2,2>()) ));
  CALL_SUBTEST(( forward_jacobian(TestFunc1<double,2,3>()) ));
  CALL_SUBTEST(( forward_jacobian(TestFunc1<double,3,2>()) ));
  CALL_SUBTEST(( forward_jacobian(TestFunc1<double,3,3>()) ));
  CALL_SUBTEST(( forward_jacobian(TestFunc1<double>(3,3)) ));
}

void test_autodiff()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( test_autodiff_scalar() );
    CALL_SUBTEST_2( test_autodiff_vector() );
    CALL_SUBTEST_3( test_autodiff_jacobian() );
  }
}

