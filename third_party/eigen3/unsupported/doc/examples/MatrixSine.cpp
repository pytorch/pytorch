#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>

using namespace Eigen;

int main()
{
  MatrixXd A = MatrixXd::Random(3,3);
  std::cout << "A = \n" << A << "\n\n";

  MatrixXd sinA = A.sin();
  std::cout << "sin(A) = \n" << sinA << "\n\n";

  MatrixXd cosA = A.cos();
  std::cout << "cos(A) = \n" << cosA << "\n\n";
  
  // The matrix functions satisfy sin^2(A) + cos^2(A) = I, 
  // like the scalar functions.
  std::cout << "sin^2(A) + cos^2(A) = \n" << sinA*sinA + cosA*cosA << "\n\n";
}
