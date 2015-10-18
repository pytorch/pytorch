// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Kolja Brix <brix@igpm.rwth-aachen.de>
// Copyright (C) 2011 Andreas Platen <andiplaten@gmx.de>
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "sparse.h"
#include <Eigen/SparseExtra>
#include <Eigen/KroneckerProduct>


template<typename MatrixType>
void check_dimension(const MatrixType& ab, const int rows,  const int cols)
{
  VERIFY_IS_EQUAL(ab.rows(), rows);
  VERIFY_IS_EQUAL(ab.cols(), cols);
}


template<typename MatrixType>
void check_kronecker_product(const MatrixType& ab)
{
  VERIFY_IS_EQUAL(ab.rows(), 6);
  VERIFY_IS_EQUAL(ab.cols(), 6);
  VERIFY_IS_EQUAL(ab.nonZeros(),  36);
  VERIFY_IS_APPROX(ab.coeff(0,0), -0.4017367630386106);
  VERIFY_IS_APPROX(ab.coeff(0,1),  0.1056863433932735);
  VERIFY_IS_APPROX(ab.coeff(0,2), -0.7255206194554212);
  VERIFY_IS_APPROX(ab.coeff(0,3),  0.1908653336744706);
  VERIFY_IS_APPROX(ab.coeff(0,4),  0.350864567234111);
  VERIFY_IS_APPROX(ab.coeff(0,5), -0.0923032108308013);
  VERIFY_IS_APPROX(ab.coeff(1,0),  0.415417514804677);
  VERIFY_IS_APPROX(ab.coeff(1,1), -0.2369227701722048);
  VERIFY_IS_APPROX(ab.coeff(1,2),  0.7502275131458511);
  VERIFY_IS_APPROX(ab.coeff(1,3), -0.4278731019742696);
  VERIFY_IS_APPROX(ab.coeff(1,4), -0.3628129162264507);
  VERIFY_IS_APPROX(ab.coeff(1,5),  0.2069210808481275);
  VERIFY_IS_APPROX(ab.coeff(2,0),  0.05465890160863986);
  VERIFY_IS_APPROX(ab.coeff(2,1), -0.2634092511419858);
  VERIFY_IS_APPROX(ab.coeff(2,2),  0.09871180285793758);
  VERIFY_IS_APPROX(ab.coeff(2,3), -0.4757066334017702);
  VERIFY_IS_APPROX(ab.coeff(2,4), -0.04773740823058334);
  VERIFY_IS_APPROX(ab.coeff(2,5),  0.2300535609645254);
  VERIFY_IS_APPROX(ab.coeff(3,0), -0.8172945853260133);
  VERIFY_IS_APPROX(ab.coeff(3,1),  0.2150086428359221);
  VERIFY_IS_APPROX(ab.coeff(3,2),  0.5825113847292743);
  VERIFY_IS_APPROX(ab.coeff(3,3), -0.1532433770097174);
  VERIFY_IS_APPROX(ab.coeff(3,4), -0.329383387282399);
  VERIFY_IS_APPROX(ab.coeff(3,5),  0.08665207912033064);
  VERIFY_IS_APPROX(ab.coeff(4,0),  0.8451267514863225);
  VERIFY_IS_APPROX(ab.coeff(4,1), -0.481996458918977);
  VERIFY_IS_APPROX(ab.coeff(4,2), -0.6023482390791535);
  VERIFY_IS_APPROX(ab.coeff(4,3),  0.3435339347164565);
  VERIFY_IS_APPROX(ab.coeff(4,4),  0.3406002157428891);
  VERIFY_IS_APPROX(ab.coeff(4,5), -0.1942526344200915);
  VERIFY_IS_APPROX(ab.coeff(5,0),  0.1111982482925399);
  VERIFY_IS_APPROX(ab.coeff(5,1), -0.5358806424754169);
  VERIFY_IS_APPROX(ab.coeff(5,2), -0.07925446559335647);
  VERIFY_IS_APPROX(ab.coeff(5,3),  0.3819388757769038);
  VERIFY_IS_APPROX(ab.coeff(5,4),  0.04481475387219876);
  VERIFY_IS_APPROX(ab.coeff(5,5), -0.2159688616158057);
}


template<typename MatrixType>
void check_sparse_kronecker_product(const MatrixType& ab)
{
  VERIFY_IS_EQUAL(ab.rows(), 12);
  VERIFY_IS_EQUAL(ab.cols(), 10);
  VERIFY_IS_EQUAL(ab.nonZeros(), 3*2);
  VERIFY_IS_APPROX(ab.coeff(3,0), -0.04);
  VERIFY_IS_APPROX(ab.coeff(5,1),  0.05);
  VERIFY_IS_APPROX(ab.coeff(0,6), -0.08);
  VERIFY_IS_APPROX(ab.coeff(2,7),  0.10);
  VERIFY_IS_APPROX(ab.coeff(6,8),  0.12);
  VERIFY_IS_APPROX(ab.coeff(8,9), -0.15);
}


void test_kronecker_product()
{
  // DM = dense matrix; SM = sparse matrix

  Matrix<double, 2, 3> DM_a;
  SparseMatrix<double> SM_a(2,3);
  SM_a.insert(0,0) = DM_a.coeffRef(0,0) = -0.4461540300782201;
  SM_a.insert(0,1) = DM_a.coeffRef(0,1) = -0.8057364375283049;
  SM_a.insert(0,2) = DM_a.coeffRef(0,2) =  0.3896572459516341;
  SM_a.insert(1,0) = DM_a.coeffRef(1,0) = -0.9076572187376921;
  SM_a.insert(1,1) = DM_a.coeffRef(1,1) =  0.6469156566545853;
  SM_a.insert(1,2) = DM_a.coeffRef(1,2) = -0.3658010398782789;
 
  MatrixXd             DM_b(3,2);
  SparseMatrix<double> SM_b(3,2);
  SM_b.insert(0,0) = DM_b.coeffRef(0,0) =  0.9004440976767099;
  SM_b.insert(0,1) = DM_b.coeffRef(0,1) = -0.2368830858139832;
  SM_b.insert(1,0) = DM_b.coeffRef(1,0) = -0.9311078389941825;
  SM_b.insert(1,1) = DM_b.coeffRef(1,1) =  0.5310335762980047;
  SM_b.insert(2,0) = DM_b.coeffRef(2,0) = -0.1225112806872035;
  SM_b.insert(2,1) = DM_b.coeffRef(2,1) =  0.5903998022741264;

  SparseMatrix<double,RowMajor> SM_row_a(SM_a), SM_row_b(SM_b);

  // test DM_fixedSize = kroneckerProduct(DM_block,DM)
  Matrix<double, 6, 6> DM_fix_ab = kroneckerProduct(DM_a.topLeftCorner<2,3>(),DM_b);

  CALL_SUBTEST(check_kronecker_product(DM_fix_ab));
  CALL_SUBTEST(check_kronecker_product(kroneckerProduct(DM_a.topLeftCorner<2,3>(),DM_b)));

  for(int i=0;i<DM_fix_ab.rows();++i)
    for(int j=0;j<DM_fix_ab.cols();++j)
       VERIFY_IS_APPROX(kroneckerProduct(DM_a,DM_b).coeff(i,j), DM_fix_ab(i,j));

  // test DM_block = kroneckerProduct(DM,DM)
  MatrixXd DM_block_ab(10,15);
  DM_block_ab.block<6,6>(2,5) = kroneckerProduct(DM_a,DM_b);
  CALL_SUBTEST(check_kronecker_product(DM_block_ab.block<6,6>(2,5)));

  // test DM = kroneckerProduct(DM,DM)
  MatrixXd DM_ab = kroneckerProduct(DM_a,DM_b);
  CALL_SUBTEST(check_kronecker_product(DM_ab));
  CALL_SUBTEST(check_kronecker_product(kroneckerProduct(DM_a,DM_b)));

  // test SM = kroneckerProduct(SM,DM)
  SparseMatrix<double> SM_ab = kroneckerProduct(SM_a,DM_b);
  CALL_SUBTEST(check_kronecker_product(SM_ab));
  SparseMatrix<double,RowMajor> SM_ab2 = kroneckerProduct(SM_a,DM_b);
  CALL_SUBTEST(check_kronecker_product(SM_ab2));
  CALL_SUBTEST(check_kronecker_product(kroneckerProduct(SM_a,DM_b)));

  // test SM = kroneckerProduct(DM,SM)
  SM_ab.setZero();
  SM_ab.insert(0,0)=37.0;
  SM_ab = kroneckerProduct(DM_a,SM_b);
  CALL_SUBTEST(check_kronecker_product(SM_ab));
  SM_ab2.setZero();
  SM_ab2.insert(0,0)=37.0;
  SM_ab2 = kroneckerProduct(DM_a,SM_b);
  CALL_SUBTEST(check_kronecker_product(SM_ab2));
  CALL_SUBTEST(check_kronecker_product(kroneckerProduct(DM_a,SM_b)));

  // test SM = kroneckerProduct(SM,SM)
  SM_ab.resize(2,33);
  SM_ab.insert(0,0)=37.0;
  SM_ab = kroneckerProduct(SM_a,SM_b);
  CALL_SUBTEST(check_kronecker_product(SM_ab));
  SM_ab2.resize(5,11);
  SM_ab2.insert(0,0)=37.0;
  SM_ab2 = kroneckerProduct(SM_a,SM_b);
  CALL_SUBTEST(check_kronecker_product(SM_ab2));
  CALL_SUBTEST(check_kronecker_product(kroneckerProduct(SM_a,SM_b)));

  // test SM = kroneckerProduct(SM,SM) with sparse pattern
  SM_a.resize(4,5);
  SM_b.resize(3,2);
  SM_a.resizeNonZeros(0);
  SM_b.resizeNonZeros(0);
  SM_a.insert(1,0) = -0.1;
  SM_a.insert(0,3) = -0.2;
  SM_a.insert(2,4) =  0.3;
  SM_a.finalize();
  
  SM_b.insert(0,0) =  0.4;
  SM_b.insert(2,1) = -0.5;
  SM_b.finalize();
  SM_ab.resize(1,1);
  SM_ab.insert(0,0)=37.0;
  SM_ab = kroneckerProduct(SM_a,SM_b);
  CALL_SUBTEST(check_sparse_kronecker_product(SM_ab));

  // test dimension of result of DM = kroneckerProduct(DM,DM)
  MatrixXd DM_a2(2,1);
  MatrixXd DM_b2(5,4);
  MatrixXd DM_ab2 = kroneckerProduct(DM_a2,DM_b2);
  CALL_SUBTEST(check_dimension(DM_ab2,2*5,1*4));
  DM_a2.resize(10,9);
  DM_b2.resize(4,8);
  DM_ab2 = kroneckerProduct(DM_a2,DM_b2);
  CALL_SUBTEST(check_dimension(DM_ab2,10*4,9*8));
  
  for(int i = 0; i < g_repeat; i++)
  {
    double density = Eigen::internal::random<double>(0.01,0.5);
    int ra = Eigen::internal::random<int>(1,50);
    int ca = Eigen::internal::random<int>(1,50);
    int rb = Eigen::internal::random<int>(1,50);
    int cb = Eigen::internal::random<int>(1,50);
    SparseMatrix<float,ColMajor> sA(ra,ca), sB(rb,cb), sC;
    SparseMatrix<float,RowMajor> sC2;
    MatrixXf dA(ra,ca), dB(rb,cb), dC;
    initSparse(density, dA, sA);
    initSparse(density, dB, sB);
    
    sC = kroneckerProduct(sA,sB);
    dC = kroneckerProduct(dA,dB);
    VERIFY_IS_APPROX(MatrixXf(sC),dC);
    
    sC = kroneckerProduct(sA.transpose(),sB);
    dC = kroneckerProduct(dA.transpose(),dB);
    VERIFY_IS_APPROX(MatrixXf(sC),dC);
    
    sC = kroneckerProduct(sA.transpose(),sB.transpose());
    dC = kroneckerProduct(dA.transpose(),dB.transpose());
    VERIFY_IS_APPROX(MatrixXf(sC),dC);
    
    sC = kroneckerProduct(sA,sB.transpose());
    dC = kroneckerProduct(dA,dB.transpose());
    VERIFY_IS_APPROX(MatrixXf(sC),dC);
    
    sC2 = kroneckerProduct(sA,sB);
    dC = kroneckerProduct(dA,dB);
    VERIFY_IS_APPROX(MatrixXf(sC2),dC);
    
    sC2 = kroneckerProduct(dA,sB);
    dC = kroneckerProduct(dA,dB);
    VERIFY_IS_APPROX(MatrixXf(sC2),dC);
    
    sC2 = kroneckerProduct(sA,dB);
    dC = kroneckerProduct(dA,dB);
    VERIFY_IS_APPROX(MatrixXf(sC2),dC);
    
    sC2 = kroneckerProduct(2*sA,sB);
    dC = kroneckerProduct(2*dA,dB);
    VERIFY_IS_APPROX(MatrixXf(sC2),dC);
  }
}
