#include <unsupported/Eigen/Polynomials>
#include <vector>
#include <iostream>

using namespace Eigen;
using namespace std;

int main()
{
  typedef Matrix<double,5,1> Vector5d;

  Vector5d roots = Vector5d::Random();
  cout << "Roots: " << roots.transpose() << endl;
  Eigen::Matrix<double,6,1> polynomial;
  roots_to_monicPolynomial( roots, polynomial );

  PolynomialSolver<double,5> psolve( polynomial );
  cout << "Complex roots: " << psolve.roots().transpose() << endl;

  std::vector<double> realRoots;
  psolve.realRoots( realRoots );
  Map<Vector5d> mapRR( &realRoots[0] );
  cout << "Real roots: " << mapRR.transpose() << endl;

  cout << endl;
  cout << "Illustration of the convergence problem with the QR algorithm: " << endl;
  cout << "---------------------------------------------------------------" << endl;
  Eigen::Matrix<float,7,1> hardCase_polynomial;
  hardCase_polynomial <<
  -0.957, 0.9219, 0.3516, 0.9453, -0.4023, -0.5508, -0.03125;
  cout << "Hard case polynomial defined by floats: " << hardCase_polynomial.transpose() << endl;
  PolynomialSolver<float,6> psolvef( hardCase_polynomial );
  cout << "Complex roots: " << psolvef.roots().transpose() << endl;
  Eigen::Matrix<float,6,1> evals;
  for( int i=0; i<6; ++i ){ evals[i] = std::abs( poly_eval( hardCase_polynomial, psolvef.roots()[i] ) ); }
  cout << "Norms of the evaluations of the polynomial at the roots: " << evals.transpose() << endl << endl;

  cout << "Using double's almost always solves the problem for small degrees: " << endl;
  cout << "-------------------------------------------------------------------" << endl;
  PolynomialSolver<double,6> psolve6d( hardCase_polynomial.cast<double>() );
  cout << "Complex roots: " << psolve6d.roots().transpose() << endl;
  for( int i=0; i<6; ++i )
  {
    std::complex<float> castedRoot( psolve6d.roots()[i].real(), psolve6d.roots()[i].imag() );
    evals[i] = std::abs( poly_eval( hardCase_polynomial, castedRoot ) );
  }
  cout << "Norms of the evaluations of the polynomial at the roots: " << evals.transpose() << endl << endl;

  cout.precision(10);
  cout << "The last root in float then in double: " << psolvef.roots()[5] << "\t" << psolve6d.roots()[5] << endl;
  std::complex<float> castedRoot( psolve6d.roots()[5].real(), psolve6d.roots()[5].imag() );
  cout << "Norm of the difference: " << std::abs( psolvef.roots()[5] - castedRoot ) << endl;
}
