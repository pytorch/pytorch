#include <unsupported/Eigen/Polynomials>
#include <iostream>

using namespace Eigen;
using namespace std;

int main()
{
  Vector4d roots = Vector4d::Random();
  cout << "Roots: " << roots.transpose() << endl;
  Eigen::Matrix<double,5,1> polynomial;
  roots_to_monicPolynomial( roots, polynomial );
  cout << "Polynomial: ";
  for( int i=0; i<4; ++i ){ cout << polynomial[i] << ".x^" << i << "+ "; }
  cout << polynomial[4] << ".x^4" << endl;
  Vector4d evaluation;
  for( int i=0; i<4; ++i ){
    evaluation[i] = poly_eval( polynomial, roots[i] ); }
  cout << "Evaluation of the polynomial at the roots: " << evaluation.transpose();
}
