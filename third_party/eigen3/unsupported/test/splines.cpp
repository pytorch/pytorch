// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010-2011 Hauke Heibel <heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <unsupported/Eigen/Splines>

namespace Eigen {
  
  // lets do some explicit instantiations and thus
  // force the compilation of all spline functions...
  template class Spline<double, 2, Dynamic>;
  template class Spline<double, 3, Dynamic>;

  template class Spline<double, 2, 2>;
  template class Spline<double, 2, 3>;
  template class Spline<double, 2, 4>;
  template class Spline<double, 2, 5>;

  template class Spline<float, 2, Dynamic>;
  template class Spline<float, 3, Dynamic>;

  template class Spline<float, 3, 2>;
  template class Spline<float, 3, 3>;
  template class Spline<float, 3, 4>;
  template class Spline<float, 3, 5>;

}

Spline<double, 2, Dynamic> closed_spline2d()
{
  RowVectorXd knots(12);
  knots << 0,
    0,
    0,
    0,
    0.867193179093898,
    1.660330955342408,
    2.605084834823134,
    3.484154586374428,
    4.252699478956276,
    4.252699478956276,
    4.252699478956276,
    4.252699478956276;

  MatrixXd ctrls(8,2);
  ctrls << -0.370967741935484,   0.236842105263158,
    -0.231401860693277,   0.442245185027632,
    0.344361228532831,   0.773369994120753,
    0.828990216203802,   0.106550882647595,
    0.407270163678382,  -1.043452922172848,
    -0.488467813584053,  -0.390098582530090,
    -0.494657189446427,   0.054804824897884,
    -0.370967741935484,   0.236842105263158;
  ctrls.transposeInPlace();

  return Spline<double, 2, Dynamic>(knots, ctrls);
}

/* create a reference spline */
Spline<double, 3, Dynamic> spline3d()
{
  RowVectorXd knots(11);
  knots << 0,
    0,
    0,
    0.118997681558377,
    0.162611735194631,
    0.498364051982143,
    0.655098003973841,
    0.679702676853675,
    1.000000000000000,
    1.000000000000000,
    1.000000000000000;

  MatrixXd ctrls(8,3);
  ctrls <<    0.959743958516081,   0.340385726666133,   0.585267750979777,
    0.223811939491137,   0.751267059305653,   0.255095115459269,
    0.505957051665142,   0.699076722656686,   0.890903252535799,
    0.959291425205444,   0.547215529963803,   0.138624442828679,
    0.149294005559057,   0.257508254123736,   0.840717255983663,
    0.254282178971531,   0.814284826068816,   0.243524968724989,
    0.929263623187228,   0.349983765984809,   0.196595250431208,
    0.251083857976031,   0.616044676146639,   0.473288848902729;
  ctrls.transposeInPlace();

  return Spline<double, 3, Dynamic>(knots, ctrls);
}

/* compares evaluations against known results */
void eval_spline3d()
{
  Spline3d spline = spline3d();

  RowVectorXd u(10);
  u << 0.351659507062997,
    0.830828627896291,
    0.585264091152724,
    0.549723608291140,
    0.917193663829810,
    0.285839018820374,
    0.757200229110721,
    0.753729094278495,
    0.380445846975357,
    0.567821640725221;

  MatrixXd pts(10,3);
  pts << 0.707620811535916,   0.510258911240815,   0.417485437023409,
    0.603422256426978,   0.529498282727551,   0.270351549348981,
    0.228364197569334,   0.423745615677815,   0.637687289287490,
    0.275556796335168,   0.350856706427970,   0.684295784598905,
    0.514519311047655,   0.525077224890754,   0.351628308305896,
    0.724152914315666,   0.574461155457304,   0.469860285484058,
    0.529365063753288,   0.613328702656816,   0.237837040141739,
    0.522469395136878,   0.619099658652895,   0.237139665242069,
    0.677357023849552,   0.480655768435853,   0.422227610314397,
    0.247046593173758,   0.380604672404750,   0.670065791405019;
  pts.transposeInPlace();

  for (int i=0; i<u.size(); ++i)
  {
    Vector3d pt = spline(u(i));
    VERIFY( (pt - pts.col(i)).norm() < 1e-14 );
  }
}

/* compares evaluations on corner cases */
void eval_spline3d_onbrks()
{
  Spline3d spline = spline3d();

  RowVectorXd u = spline.knots();

  MatrixXd pts(11,3);
  pts <<    0.959743958516081,   0.340385726666133,   0.585267750979777,
    0.959743958516081,   0.340385726666133,   0.585267750979777,
    0.959743958516081,   0.340385726666133,   0.585267750979777,
    0.430282980289940,   0.713074680056118,   0.720373307943349,
    0.558074875553060,   0.681617921034459,   0.804417124839942,
    0.407076008291750,   0.349707710518163,   0.617275937419545,
    0.240037008286602,   0.738739390398014,   0.324554153129411,
    0.302434111480572,   0.781162443963899,   0.240177089094644,
    0.251083857976031,   0.616044676146639,   0.473288848902729,
    0.251083857976031,   0.616044676146639,   0.473288848902729,
    0.251083857976031,   0.616044676146639,   0.473288848902729;
  pts.transposeInPlace();

  for (int i=0; i<u.size(); ++i)
  {
    Vector3d pt = spline(u(i));
    VERIFY( (pt - pts.col(i)).norm() < 1e-14 );
  }
}

void eval_closed_spline2d()
{
  Spline2d spline = closed_spline2d();

  RowVectorXd u(12);
  u << 0,
    0.332457030395796,
    0.356467130532952,
    0.453562180176215,
    0.648017921874804,
    0.973770235555003,
    1.882577647219307,
    2.289408593930498,
    3.511951429883045,
    3.884149321369450,
    4.236261590369414,
    4.252699478956276;

  MatrixXd pts(12,2);
  pts << -0.370967741935484,   0.236842105263158,
    -0.152576775123250,   0.448975001279334,
    -0.133417538277668,   0.461615613865667,
    -0.053199060826740,   0.507630360006299,
    0.114249591147281,   0.570414135097409,
    0.377810316891987,   0.560497102875315,
    0.665052120135908,  -0.157557441109611,
    0.516006487053228,  -0.559763292174825,
    -0.379486035348887,  -0.331959640488223,
    -0.462034726249078,  -0.039105670080824,
    -0.378730600917982,   0.225127015099919,
    -0.370967741935484,   0.236842105263158;
  pts.transposeInPlace();

  for (int i=0; i<u.size(); ++i)
  {
    Vector2d pt = spline(u(i));
    VERIFY( (pt - pts.col(i)).norm() < 1e-14 );
  }
}

void check_global_interpolation2d()
{
  typedef Spline2d::PointType PointType;
  typedef Spline2d::KnotVectorType KnotVectorType;
  typedef Spline2d::ControlPointVectorType ControlPointVectorType;

  ControlPointVectorType points = ControlPointVectorType::Random(2,100);

  KnotVectorType chord_lengths; // knot parameters
  Eigen::ChordLengths(points, chord_lengths);

  // interpolation without knot parameters
  {
    const Spline2d spline = SplineFitting<Spline2d>::Interpolate(points,3);  

    for (Eigen::DenseIndex i=0; i<points.cols(); ++i)
    {
      PointType pt = spline( chord_lengths(i) );
      PointType ref = points.col(i);
      VERIFY( (pt - ref).matrix().norm() < 1e-14 );
    }
  }

  // interpolation with given knot parameters
  {
    const Spline2d spline = SplineFitting<Spline2d>::Interpolate(points,3,chord_lengths);  

    for (Eigen::DenseIndex i=0; i<points.cols(); ++i)
    {
      PointType pt = spline( chord_lengths(i) );
      PointType ref = points.col(i);
      VERIFY( (pt - ref).matrix().norm() < 1e-14 );
    }
  }
}

void check_global_interpolation_with_derivatives2d()
{
  typedef Spline2d::PointType PointType;
  typedef Spline2d::KnotVectorType KnotVectorType;

  const unsigned int numPoints = 100;
  const unsigned int dimension = 2;
  const unsigned int degree = 3;

  ArrayXXd points = ArrayXXd::Random(dimension, numPoints);

  KnotVectorType knots;
  Eigen::ChordLengths(points, knots);

  ArrayXXd derivatives = ArrayXXd::Random(dimension, numPoints);
  VectorXd derivativeIndices(numPoints);

  for (Eigen::DenseIndex i = 0; i < numPoints; ++i)
      derivativeIndices(i) = static_cast<double>(i);

  const Spline2d spline = SplineFitting<Spline2d>::InterpolateWithDerivatives(
    points, derivatives, derivativeIndices, degree);  
    
  for (Eigen::DenseIndex i = 0; i < points.cols(); ++i)
  {
    PointType point = spline(knots(i));
    PointType referencePoint = points.col(i);
    VERIFY_IS_APPROX(point, referencePoint);
    PointType derivative = spline.derivatives(knots(i), 1).col(1);
    PointType referenceDerivative = derivatives.col(i);
    VERIFY_IS_APPROX(derivative, referenceDerivative);
  }
}

void test_splines()
{
  for (int i = 0; i < g_repeat; ++i)
  {
    CALL_SUBTEST( eval_spline3d() );
    CALL_SUBTEST( eval_spline3d_onbrks() );
    CALL_SUBTEST( eval_closed_spline2d() );
    CALL_SUBTEST( check_global_interpolation2d() );
    CALL_SUBTEST( check_global_interpolation_with_derivatives2d() );
  }
}
