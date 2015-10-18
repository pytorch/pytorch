#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>

using namespace Eigen;

int main()
{
  Matrix4cd A = Matrix4cd::Random();
  MatrixPower<Matrix4cd> Apow(A);

  std::cout << "The matrix A is:\n" << A << "\n\n"
	       "A^3.1 is:\n" << Apow(3.1) << "\n\n"
	       "A^3.3 is:\n" << Apow(3.3) << "\n\n"
	       "A^3.7 is:\n" << Apow(3.7) << "\n\n"
	       "A^3.9 is:\n" << Apow(3.9) << std::endl;
  return 0;
}
