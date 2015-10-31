#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>

using namespace Eigen;

int main()
{
  const double pi = std::acos(-1.0);
  Matrix3d A;
  A << cos(1), -sin(1), 0,
       sin(1),  cos(1), 0,
	   0 ,      0 , 1;
  std::cout << "The matrix A is:\n" << A << "\n\n"
	       "The matrix power A^(pi/4) is:\n" << A.pow(pi/4) << std::endl;
  return 0;
}
