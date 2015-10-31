// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Ilya Baran <ibaran@mit.edu>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <unsupported/Eigen/BVH>

namespace Eigen {

template<typename Scalar, int Dim> AlignedBox<Scalar, Dim> bounding_box(const Matrix<Scalar, Dim, 1> &v) { return AlignedBox<Scalar, Dim>(v); }

}


template<int Dim>
struct Ball
{
EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF_VECTORIZABLE_FIXED_SIZE(double, Dim)

  typedef Matrix<double, Dim, 1> VectorType;

  Ball() {}
  Ball(const VectorType &c, double r) : center(c), radius(r) {}

  VectorType center;
  double radius;
};
template<int Dim> AlignedBox<double, Dim> bounding_box(const Ball<Dim> &b)
{ return AlignedBox<double, Dim>(b.center.array() - b.radius, b.center.array() + b.radius); }

inline double SQR(double x) { return x * x; }

template<int Dim>
struct BallPointStuff //this class provides functions to be both an intersector and a minimizer, both for a ball and a point and for two trees
{
  typedef double Scalar;
  typedef Matrix<double, Dim, 1> VectorType;
  typedef Ball<Dim> BallType;
  typedef AlignedBox<double, Dim> BoxType;

  BallPointStuff() : calls(0), count(0) {}
  BallPointStuff(const VectorType &inP) : p(inP), calls(0), count(0) {}


  bool intersectVolume(const BoxType &r) { ++calls; return r.contains(p); }
  bool intersectObject(const BallType &b) {
    ++calls;
    if((b.center - p).squaredNorm() < SQR(b.radius))
      ++count;
    return false; //continue
  }

  bool intersectVolumeVolume(const BoxType &r1, const BoxType &r2) { ++calls; return !(r1.intersection(r2)).isNull(); }
  bool intersectVolumeObject(const BoxType &r, const BallType &b) { ++calls; return r.squaredExteriorDistance(b.center) < SQR(b.radius); }
  bool intersectObjectVolume(const BallType &b, const BoxType &r) { ++calls; return r.squaredExteriorDistance(b.center) < SQR(b.radius); }
  bool intersectObjectObject(const BallType &b1, const BallType &b2){
    ++calls;
    if((b1.center - b2.center).norm() < b1.radius + b2.radius)
      ++count;
    return false;
  }
  bool intersectVolumeObject(const BoxType &r, const VectorType &v) { ++calls; return r.contains(v); }
  bool intersectObjectObject(const BallType &b, const VectorType &v){
    ++calls;
    if((b.center - v).squaredNorm() < SQR(b.radius))
      ++count;
    return false;
  }

  double minimumOnVolume(const BoxType &r) { ++calls; return r.squaredExteriorDistance(p); }
  double minimumOnObject(const BallType &b) { ++calls; return (std::max)(0., (b.center - p).squaredNorm() - SQR(b.radius)); }
  double minimumOnVolumeVolume(const BoxType &r1, const BoxType &r2) { ++calls; return r1.squaredExteriorDistance(r2); }
  double minimumOnVolumeObject(const BoxType &r, const BallType &b) { ++calls; return SQR((std::max)(0., r.exteriorDistance(b.center) - b.radius)); }
  double minimumOnObjectVolume(const BallType &b, const BoxType &r) { ++calls; return SQR((std::max)(0., r.exteriorDistance(b.center) - b.radius)); }
  double minimumOnObjectObject(const BallType &b1, const BallType &b2){ ++calls; return SQR((std::max)(0., (b1.center - b2.center).norm() - b1.radius - b2.radius)); }
  double minimumOnVolumeObject(const BoxType &r, const VectorType &v) { ++calls; return r.squaredExteriorDistance(v); }
  double minimumOnObjectObject(const BallType &b, const VectorType &v){ ++calls; return SQR((std::max)(0., (b.center - v).norm() - b.radius)); }

  VectorType p;
  int calls;
  int count;
};


template<int Dim>
struct TreeTest
{
  typedef Matrix<double, Dim, 1> VectorType;
  typedef std::vector<VectorType, aligned_allocator<VectorType> > VectorTypeList;
  typedef Ball<Dim> BallType;
  typedef std::vector<BallType, aligned_allocator<BallType> > BallTypeList;
  typedef AlignedBox<double, Dim> BoxType;

  void testIntersect1()
  {
    BallTypeList b;
    for(int i = 0; i < 500; ++i) {
        b.push_back(BallType(VectorType::Random(), 0.5 * internal::random(0., 1.)));
    }
    KdBVH<double, Dim, BallType> tree(b.begin(), b.end());

    VectorType pt = VectorType::Random();
    BallPointStuff<Dim> i1(pt), i2(pt);

    for(int i = 0; i < (int)b.size(); ++i)
      i1.intersectObject(b[i]);

    BVIntersect(tree, i2);

    VERIFY(i1.count == i2.count);
  }

  void testMinimize1()
  {
    BallTypeList b;
    for(int i = 0; i < 500; ++i) {
        b.push_back(BallType(VectorType::Random(), 0.01 * internal::random(0., 1.)));
    }
    KdBVH<double, Dim, BallType> tree(b.begin(), b.end());

    VectorType pt = VectorType::Random();
    BallPointStuff<Dim> i1(pt), i2(pt);

    double m1 = (std::numeric_limits<double>::max)(), m2 = m1;

    for(int i = 0; i < (int)b.size(); ++i)
      m1 = (std::min)(m1, i1.minimumOnObject(b[i]));

    m2 = BVMinimize(tree, i2);

    VERIFY_IS_APPROX(m1, m2);
  }

  void testIntersect2()
  {
    BallTypeList b;
    VectorTypeList v;

    for(int i = 0; i < 50; ++i) {
        b.push_back(BallType(VectorType::Random(), 0.5 * internal::random(0., 1.)));
        for(int j = 0; j < 3; ++j)
            v.push_back(VectorType::Random());
    }

    KdBVH<double, Dim, BallType> tree(b.begin(), b.end());
    KdBVH<double, Dim, VectorType> vTree(v.begin(), v.end());

    BallPointStuff<Dim> i1, i2;

    for(int i = 0; i < (int)b.size(); ++i)
        for(int j = 0; j < (int)v.size(); ++j)
            i1.intersectObjectObject(b[i], v[j]);

    BVIntersect(tree, vTree, i2);

    VERIFY(i1.count == i2.count);
  }

  void testMinimize2()
  {
    BallTypeList b;
    VectorTypeList v;

    for(int i = 0; i < 50; ++i) {
        b.push_back(BallType(VectorType::Random(), 1e-7 + 1e-6 * internal::random(0., 1.)));
        for(int j = 0; j < 3; ++j)
            v.push_back(VectorType::Random());
    }

    KdBVH<double, Dim, BallType> tree(b.begin(), b.end());
    KdBVH<double, Dim, VectorType> vTree(v.begin(), v.end());

    BallPointStuff<Dim> i1, i2;

    double m1 = (std::numeric_limits<double>::max)(), m2 = m1;

    for(int i = 0; i < (int)b.size(); ++i)
        for(int j = 0; j < (int)v.size(); ++j)
            m1 = (std::min)(m1, i1.minimumOnObjectObject(b[i], v[j]));

    m2 = BVMinimize(tree, vTree, i2);

    VERIFY_IS_APPROX(m1, m2);
  }
};


void test_BVH()
{
  for(int i = 0; i < g_repeat; i++) {
#ifdef EIGEN_TEST_PART_1
    TreeTest<2> test2;
    CALL_SUBTEST(test2.testIntersect1());
    CALL_SUBTEST(test2.testMinimize1());
    CALL_SUBTEST(test2.testIntersect2());
    CALL_SUBTEST(test2.testMinimize2());
#endif

#ifdef EIGEN_TEST_PART_2
    TreeTest<3> test3;
    CALL_SUBTEST(test3.testIntersect1());
    CALL_SUBTEST(test3.testMinimize1());
    CALL_SUBTEST(test3.testIntersect2());
    CALL_SUBTEST(test3.testMinimize2());
#endif

#ifdef EIGEN_TEST_PART_3
    TreeTest<4> test4;
    CALL_SUBTEST(test4.testIntersect1());
    CALL_SUBTEST(test4.testMinimize1());
    CALL_SUBTEST(test4.testIntersect2());
    CALL_SUBTEST(test4.testMinimize2());
#endif
  }
}
