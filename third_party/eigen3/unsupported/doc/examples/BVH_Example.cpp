#include <Eigen/StdVector>
#include <unsupported/Eigen/BVH>
#include <iostream>

using namespace Eigen;
typedef AlignedBox<double, 2> Box2d;

namespace Eigen {
    namespace internal {
        Box2d bounding_box(const Vector2d &v) { return Box2d(v, v); } //compute the bounding box of a single point
    }
}

struct PointPointMinimizer //how to compute squared distances between points and rectangles
{
  PointPointMinimizer() : calls(0) {}
  typedef double Scalar;

  double minimumOnVolumeVolume(const Box2d &r1, const Box2d &r2) { ++calls; return r1.squaredExteriorDistance(r2); }
  double minimumOnVolumeObject(const Box2d &r, const Vector2d &v) { ++calls; return r.squaredExteriorDistance(v); }
  double minimumOnObjectVolume(const Vector2d &v, const Box2d &r) { ++calls; return r.squaredExteriorDistance(v); }
  double minimumOnObjectObject(const Vector2d &v1, const Vector2d &v2) { ++calls; return (v1 - v2).squaredNorm(); }

  int calls;
};

int main()
{
  typedef std::vector<Vector2d, aligned_allocator<Vector2d> > StdVectorOfVector2d;
  StdVectorOfVector2d redPoints, bluePoints;
  for(int i = 0; i < 100; ++i) { //initialize random set of red points and blue points
    redPoints.push_back(Vector2d::Random());
    bluePoints.push_back(Vector2d::Random());
  }

  PointPointMinimizer minimizer;
  double minDistSq = std::numeric_limits<double>::max();

  //brute force to find closest red-blue pair
  for(int i = 0; i < (int)redPoints.size(); ++i)
    for(int j = 0; j < (int)bluePoints.size(); ++j)
      minDistSq = std::min(minDistSq, minimizer.minimumOnObjectObject(redPoints[i], bluePoints[j]));
  std::cout << "Brute force distance = " << sqrt(minDistSq) << ", calls = " << minimizer.calls << std::endl;

  //using BVH to find closest red-blue pair
  minimizer.calls = 0;
  KdBVH<double, 2, Vector2d> redTree(redPoints.begin(), redPoints.end()), blueTree(bluePoints.begin(), bluePoints.end()); //construct the trees
  minDistSq = BVMinimize(redTree, blueTree, minimizer); //actual BVH minimization call
  std::cout << "BVH distance         = " << sqrt(minDistSq) << ", calls = " << minimizer.calls << std::endl;

  return 0;
}
