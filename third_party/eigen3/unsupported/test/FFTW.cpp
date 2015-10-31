// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Mark Borgerding mark a borgerding net
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <unsupported/Eigen/FFT>

template <typename T> 
std::complex<T> RandomCpx() { return std::complex<T>( (T)(rand()/(T)RAND_MAX - .5), (T)(rand()/(T)RAND_MAX - .5) ); }

using namespace std;
using namespace Eigen;


template < typename T>
complex<long double>  promote(complex<T> x) { return complex<long double>(x.real(),x.imag()); }

complex<long double>  promote(float x) { return complex<long double>( x); }
complex<long double>  promote(double x) { return complex<long double>( x); }
complex<long double>  promote(long double x) { return complex<long double>( x); }
    

    template <typename VT1,typename VT2>
    long double fft_rmse( const VT1 & fftbuf,const VT2 & timebuf)
    {
        long double totalpower=0;
        long double difpower=0;
        long double pi = acos((long double)-1 );
        for (size_t k0=0;k0<(size_t)fftbuf.size();++k0) {
            complex<long double> acc = 0;
            long double phinc = -2.*k0* pi / timebuf.size();
            for (size_t k1=0;k1<(size_t)timebuf.size();++k1) {
                acc +=  promote( timebuf[k1] ) * exp( complex<long double>(0,k1*phinc) );
            }
            totalpower += numext::abs2(acc);
            complex<long double> x = promote(fftbuf[k0]); 
            complex<long double> dif = acc - x;
            difpower += numext::abs2(dif);
            //cerr << k0 << "\t" << acc << "\t" <<  x << "\t" << sqrt(numext::abs2(dif)) << endl;
        }
        cerr << "rmse:" << sqrt(difpower/totalpower) << endl;
        return sqrt(difpower/totalpower);
    }

    template <typename VT1,typename VT2>
    long double dif_rmse( const VT1 buf1,const VT2 buf2)
    {
        long double totalpower=0;
        long double difpower=0;
        size_t n = (min)( buf1.size(),buf2.size() );
        for (size_t k=0;k<n;++k) {
            totalpower += (numext::abs2( buf1[k] ) + numext::abs2(buf2[k]) )/2.;
            difpower += numext::abs2(buf1[k] - buf2[k]);
        }
        return sqrt(difpower/totalpower);
    }

enum { StdVectorContainer, EigenVectorContainer };

template<int Container, typename Scalar> struct VectorType;

template<typename Scalar> struct VectorType<StdVectorContainer,Scalar>
{
  typedef vector<Scalar> type;
};

template<typename Scalar> struct VectorType<EigenVectorContainer,Scalar>
{
  typedef Matrix<Scalar,Dynamic,1> type;
};

template <int Container, typename T>
void test_scalar_generic(int nfft)
{
    typedef typename FFT<T>::Complex Complex;
    typedef typename FFT<T>::Scalar Scalar;
    typedef typename VectorType<Container,Scalar>::type ScalarVector;
    typedef typename VectorType<Container,Complex>::type ComplexVector;

    FFT<T> fft;
    ScalarVector tbuf(nfft);
    ComplexVector freqBuf;
    for (int k=0;k<nfft;++k)
        tbuf[k]= (T)( rand()/(double)RAND_MAX - .5);

    // make sure it DOESN'T give the right full spectrum answer
    // if we've asked for half-spectrum
    fft.SetFlag(fft.HalfSpectrum );
    fft.fwd( freqBuf,tbuf);
    VERIFY((size_t)freqBuf.size() == (size_t)( (nfft>>1)+1) );
    VERIFY( fft_rmse(freqBuf,tbuf) < test_precision<T>()  );// gross check

    fft.ClearFlag(fft.HalfSpectrum );
    fft.fwd( freqBuf,tbuf);
    VERIFY( (size_t)freqBuf.size() == (size_t)nfft);
    VERIFY( fft_rmse(freqBuf,tbuf) < test_precision<T>()  );// gross check

    if (nfft&1)
        return; // odd FFTs get the wrong size inverse FFT

    ScalarVector tbuf2;
    fft.inv( tbuf2 , freqBuf);
    VERIFY( dif_rmse(tbuf,tbuf2) < test_precision<T>()  );// gross check


    // verify that the Unscaled flag takes effect
    ScalarVector tbuf3;
    fft.SetFlag(fft.Unscaled);

    fft.inv( tbuf3 , freqBuf);

    for (int k=0;k<nfft;++k)
        tbuf3[k] *= T(1./nfft);


    //for (size_t i=0;i<(size_t) tbuf.size();++i)
    //    cout << "freqBuf=" << freqBuf[i] << " in2=" << tbuf3[i] << " -  in=" << tbuf[i] << " => " << (tbuf3[i] - tbuf[i] ) <<  endl;

    VERIFY( dif_rmse(tbuf,tbuf3) < test_precision<T>()  );// gross check

    // verify that ClearFlag works
    fft.ClearFlag(fft.Unscaled);
    fft.inv( tbuf2 , freqBuf);
    VERIFY( dif_rmse(tbuf,tbuf2) < test_precision<T>()  );// gross check
}

template <typename T>
void test_scalar(int nfft)
{
  test_scalar_generic<StdVectorContainer,T>(nfft);
  //test_scalar_generic<EigenVectorContainer,T>(nfft);
}


template <int Container, typename T>
void test_complex_generic(int nfft)
{
    typedef typename FFT<T>::Complex Complex;
    typedef typename VectorType<Container,Complex>::type ComplexVector;

    FFT<T> fft;

    ComplexVector inbuf(nfft);
    ComplexVector outbuf;
    ComplexVector buf3;
    for (int k=0;k<nfft;++k)
        inbuf[k]= Complex( (T)(rand()/(double)RAND_MAX - .5), (T)(rand()/(double)RAND_MAX - .5) );
    fft.fwd( outbuf , inbuf);

    VERIFY( fft_rmse(outbuf,inbuf) < test_precision<T>()  );// gross check
    fft.inv( buf3 , outbuf);

    VERIFY( dif_rmse(inbuf,buf3) < test_precision<T>()  );// gross check

    // verify that the Unscaled flag takes effect
    ComplexVector buf4;
    fft.SetFlag(fft.Unscaled);
    fft.inv( buf4 , outbuf);
    for (int k=0;k<nfft;++k)
        buf4[k] *= T(1./nfft);
    VERIFY( dif_rmse(inbuf,buf4) < test_precision<T>()  );// gross check

    // verify that ClearFlag works
    fft.ClearFlag(fft.Unscaled);
    fft.inv( buf3 , outbuf);
    VERIFY( dif_rmse(inbuf,buf3) < test_precision<T>()  );// gross check
}

template <typename T>
void test_complex(int nfft)
{
  test_complex_generic<StdVectorContainer,T>(nfft);
  test_complex_generic<EigenVectorContainer,T>(nfft);
}
/*
template <typename T,int nrows,int ncols>
void test_complex2d()
{
    typedef typename Eigen::FFT<T>::Complex Complex;
    FFT<T> fft;
    Eigen::Matrix<Complex,nrows,ncols> src,src2,dst,dst2;

    src = Eigen::Matrix<Complex,nrows,ncols>::Random();
    //src =  Eigen::Matrix<Complex,nrows,ncols>::Identity();

    for (int k=0;k<ncols;k++) {
        Eigen::Matrix<Complex,nrows,1> tmpOut;
        fft.fwd( tmpOut,src.col(k) );
        dst2.col(k) = tmpOut;
    }

    for (int k=0;k<nrows;k++) {
        Eigen::Matrix<Complex,1,ncols> tmpOut;
        fft.fwd( tmpOut,  dst2.row(k) );
        dst2.row(k) = tmpOut;
    }

    fft.fwd2(dst.data(),src.data(),ncols,nrows);
    fft.inv2(src2.data(),dst.data(),ncols,nrows);
    VERIFY( (src-src2).norm() < test_precision<T>() );
    VERIFY( (dst-dst2).norm() < test_precision<T>() );
}
*/


void test_return_by_value(int len)
{
    VectorXf in;
    VectorXf in1;
    in.setRandom( len );
    VectorXcf out1,out2;
    FFT<float> fft;

    fft.SetFlag(fft.HalfSpectrum );

    fft.fwd(out1,in);
    out2 = fft.fwd(in);
    VERIFY( (out1-out2).norm() < test_precision<float>() );
    in1 = fft.inv(out1);
    VERIFY( (in1-in).norm() < test_precision<float>() );
}

void test_FFTW()
{
  CALL_SUBTEST( test_return_by_value(32) );
  //CALL_SUBTEST( ( test_complex2d<float,4,8> () ) ); CALL_SUBTEST( ( test_complex2d<double,4,8> () ) );
  //CALL_SUBTEST( ( test_complex2d<long double,4,8> () ) );
  CALL_SUBTEST( test_complex<float>(32) ); CALL_SUBTEST( test_complex<double>(32) ); 
  CALL_SUBTEST( test_complex<float>(256) ); CALL_SUBTEST( test_complex<double>(256) ); 
  CALL_SUBTEST( test_complex<float>(3*8) ); CALL_SUBTEST( test_complex<double>(3*8) ); 
  CALL_SUBTEST( test_complex<float>(5*32) ); CALL_SUBTEST( test_complex<double>(5*32) ); 
  CALL_SUBTEST( test_complex<float>(2*3*4) ); CALL_SUBTEST( test_complex<double>(2*3*4) ); 
  CALL_SUBTEST( test_complex<float>(2*3*4*5) ); CALL_SUBTEST( test_complex<double>(2*3*4*5) ); 
  CALL_SUBTEST( test_complex<float>(2*3*4*5*7) ); CALL_SUBTEST( test_complex<double>(2*3*4*5*7) ); 

  CALL_SUBTEST( test_scalar<float>(32) ); CALL_SUBTEST( test_scalar<double>(32) ); 
  CALL_SUBTEST( test_scalar<float>(45) ); CALL_SUBTEST( test_scalar<double>(45) ); 
  CALL_SUBTEST( test_scalar<float>(50) ); CALL_SUBTEST( test_scalar<double>(50) ); 
  CALL_SUBTEST( test_scalar<float>(256) ); CALL_SUBTEST( test_scalar<double>(256) ); 
  CALL_SUBTEST( test_scalar<float>(2*3*4*5*7) ); CALL_SUBTEST( test_scalar<double>(2*3*4*5*7) ); 
  
  #ifdef EIGEN_HAS_FFTWL
  CALL_SUBTEST( test_complex<long double>(32) );
  CALL_SUBTEST( test_complex<long double>(256) );
  CALL_SUBTEST( test_complex<long double>(3*8) );
  CALL_SUBTEST( test_complex<long double>(5*32) );
  CALL_SUBTEST( test_complex<long double>(2*3*4) );
  CALL_SUBTEST( test_complex<long double>(2*3*4*5) );
  CALL_SUBTEST( test_complex<long double>(2*3*4*5*7) );
  
  CALL_SUBTEST( test_scalar<long double>(32) );
  CALL_SUBTEST( test_scalar<long double>(45) );
  CALL_SUBTEST( test_scalar<long double>(50) );
  CALL_SUBTEST( test_scalar<long double>(256) );
  CALL_SUBTEST( test_scalar<long double>(2*3*4*5*7) );
  #endif
}
