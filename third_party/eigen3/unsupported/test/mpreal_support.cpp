#include "main.h"
#include <Eigen/MPRealSupport>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>
#include <sstream>

using namespace mpfr;
using namespace Eigen;

void test_mpreal_support()
{
  // set precision to 256 bits (double has only 53 bits)
  mpreal::set_default_prec(256);
  typedef Matrix<mpreal,Eigen::Dynamic,Eigen::Dynamic> MatrixXmp;

  std::cerr << "epsilon =         " << NumTraits<mpreal>::epsilon() << "\n";
  std::cerr << "dummy_precision = " << NumTraits<mpreal>::dummy_precision() << "\n";
  std::cerr << "highest =         " << NumTraits<mpreal>::highest() << "\n";
  std::cerr << "lowest =          " << NumTraits<mpreal>::lowest() << "\n";

  for(int i = 0; i < g_repeat; i++) {
    int s = Eigen::internal::random<int>(1,100);
    MatrixXmp A = MatrixXmp::Random(s,s);
    MatrixXmp B = MatrixXmp::Random(s,s);
    MatrixXmp S = A.adjoint() * A;
    MatrixXmp X;
    
    // Basic stuffs
    VERIFY_IS_APPROX(A.real(), A);
    VERIFY(Eigen::internal::isApprox(A.array().abs2().sum(), A.squaredNorm()));
    VERIFY_IS_APPROX(A.array().exp(),         exp(A.array()));
    VERIFY_IS_APPROX(A.array().abs2().sqrt(), A.array().abs());
    VERIFY_IS_APPROX(A.array().sin(),         sin(A.array()));
    VERIFY_IS_APPROX(A.array().cos(),         cos(A.array()));

    // Cholesky
    X = S.selfadjointView<Lower>().llt().solve(B);
    VERIFY_IS_APPROX((S.selfadjointView<Lower>()*X).eval(),B);
    
    // partial LU
    X = A.lu().solve(B);
    VERIFY_IS_APPROX((A*X).eval(),B);

    // symmetric eigenvalues
    SelfAdjointEigenSolver<MatrixXmp> eig(S);
    VERIFY_IS_EQUAL(eig.info(), Success);
    VERIFY( (S.selfadjointView<Lower>() * eig.eigenvectors()).isApprox(eig.eigenvectors() * eig.eigenvalues().asDiagonal(), NumTraits<mpreal>::dummy_precision()*1e3) );
  }
  
  {
    MatrixXmp A(8,3); A.setRandom();
    // test output (interesting things happen in this code)
    std::stringstream stream;
    stream << A;
  }
}
