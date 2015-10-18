// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Gauthier Brun <brun.gauthier@gmail.com>
// Copyright (C) 2013 Nicolas Carre <nicolas.carre@ensimag.fr>
// Copyright (C) 2013 Jean Ceccato <jean.ceccato@ensimag.fr>
// Copyright (C) 2013 Pierre Zoppitelli <pierre.zoppitelli@ensimag.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// Bench to compare the efficiency of SVD algorithms

#include <iostream>
#include <bench/BenchTimer.h>
#include <unsupported/Eigen/SVD>


using namespace Eigen;
using namespace std;

// number of computations of each algorithm before the print of the time
#ifndef REPEAT
#define REPEAT 10
#endif

// number of tests of the same type
#ifndef NUMBER_SAMPLE
#define NUMBER_SAMPLE 2
#endif

template<typename MatrixType>
void bench_svd(const MatrixType& a = MatrixType())
{
  MatrixType m = MatrixType::Random(a.rows(), a.cols());
  BenchTimer timerJacobi;
  BenchTimer timerBDC;
  timerJacobi.reset();
  timerBDC.reset();

  cout << " Only compute Singular Values" <<endl;
  for (int k=1; k<=NUMBER_SAMPLE; ++k)
  {
    timerBDC.start();
    for (int i=0; i<REPEAT; ++i) 
    {
      BDCSVD<MatrixType> bdc_matrix(m);
    }
    timerBDC.stop();
    
    timerJacobi.start();
    for (int i=0; i<REPEAT; ++i) 
    {
      JacobiSVD<MatrixType> jacobi_matrix(m);
    }
    timerJacobi.stop();


    cout << "Sample " << k << " : " << REPEAT << " computations :  Jacobi : " << fixed << timerJacobi.value() << "s ";
    cout << " || " << " BDC : " << timerBDC.value() << "s " <<endl <<endl;
      
    if (timerBDC.value() >= timerJacobi.value())  
      cout << "KO : BDC is " <<  timerJacobi.value() / timerBDC.value() << "  times faster than Jacobi" <<endl;
    else 
      cout << "OK : BDC is " << timerJacobi.value() / timerBDC.value() << "  times faster than Jacobi"  <<endl;
      
  }
  cout << "       =================" <<endl;
  std::cout<< std::endl;
  timerJacobi.reset();
  timerBDC.reset();
  cout << " Computes rotaion matrix" <<endl;
  for (int k=1; k<=NUMBER_SAMPLE; ++k)
  {
    timerBDC.start();
    for (int i=0; i<REPEAT; ++i) 
    {
      BDCSVD<MatrixType> bdc_matrix(m, ComputeFullU|ComputeFullV);
    }
    timerBDC.stop();
    
    timerJacobi.start();
    for (int i=0; i<REPEAT; ++i) 
    {
      JacobiSVD<MatrixType> jacobi_matrix(m, ComputeFullU|ComputeFullV);
    }
    timerJacobi.stop();


    cout << "Sample " << k << " : " << REPEAT << " computations :  Jacobi : " << fixed << timerJacobi.value() << "s ";
    cout << " || " << " BDC : " << timerBDC.value() << "s " <<endl <<endl;
      
    if (timerBDC.value() >= timerJacobi.value())  
      cout << "KO : BDC is " <<  timerJacobi.value() / timerBDC.value() << "  times faster than Jacobi" <<endl;
    else 
      cout << "OK : BDC is " << timerJacobi.value() / timerBDC.value() << "  times faster than Jacobi"  <<endl;
      
  }
  std::cout<< std::endl;
}



int main(int argc, char* argv[])
{
  std::cout<< std::endl;

  std::cout<<"On a (Dynamic, Dynamic) (6, 6) Matrix" <<std::endl;
  bench_svd<Matrix<double,Dynamic,Dynamic> >(Matrix<double,Dynamic,Dynamic>(6, 6));
  
  std::cout<<"On a (Dynamic, Dynamic) (32, 32) Matrix" <<std::endl;
  bench_svd<Matrix<double,Dynamic,Dynamic> >(Matrix<double,Dynamic,Dynamic>(32, 32));

  //std::cout<<"On a (Dynamic, Dynamic) (128, 128) Matrix" <<std::endl;
  //bench_svd<Matrix<double,Dynamic,Dynamic> >(Matrix<double,Dynamic,Dynamic>(128, 128));

  std::cout<<"On a (Dynamic, Dynamic) (160, 160) Matrix" <<std::endl;
  bench_svd<Matrix<double,Dynamic,Dynamic> >(Matrix<double,Dynamic,Dynamic>(160, 160));
  
  std::cout<< "--------------------------------------------------------------------"<< std::endl;
           
}
