/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include "matrix_funcs.h"
#ifdef NUMPY_INTERFACE
#include <Python.h>
#include <arrayobject.h>
#endif
#include <limits>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>

extern "C" {
// #include <cblas.h>
#include "caffe2/utils/cblas.h"
}

#ifdef DOUBLE_PRECISION
#define CBLAS_GEMM cblas_dgemm
#define CBLAS_SCAL cblas_dscal
#define CBLAS_AXPY cblas_daxpy
#else
#define CBLAS_GEMM cblas_sgemm
#define CBLAS_SCAL cblas_sscal
#define CBLAS_AXPY cblas_saxpy
#endif /* DOUBLE_PRECISION */

#define MTYPE_MAX numeric_limits<MTYPE>::max()

typedef long long int int64;

class Matrix {
private:
    MTYPE* _data;
    bool _ownsData;
    int64 _numRows, _numCols;
    int64 _numElements;
    CBLAS_TRANSPOSE _trans;

    void _init(MTYPE* data, int64 numRows, int64 numCols, bool transpose, bool ownsData);
    void _tileTo2(Matrix& target) const;
    void _copyAllTo(Matrix& target) const;
    MTYPE _sum_column(int64 col) const;
    MTYPE _sum_row(int64 row) const;
    MTYPE _aggregate(MTYPE(*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const;
    void _aggregate(int64 axis, Matrix& target, MTYPE(*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const;
    MTYPE _aggregateRow(int64 row, MTYPE(*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const;
    MTYPE _aggregateCol(int64 row, MTYPE(*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const;
    void _updateDims(int64 numRows, int64 numCols);
    void _applyLoop(MTYPE(*func)(MTYPE));
    void _applyLoop(MTYPE (*func)(MTYPE), Matrix& target);
    void _applyLoop2(const Matrix& a, MTYPE(*func)(MTYPE, MTYPE), Matrix& target) const;
    void _applyLoop2(const Matrix& a, MTYPE (*func)(MTYPE,MTYPE, MTYPE), MTYPE scalar, Matrix& target) const;
    void _applyLoop2(const Matrix& a, MTYPE (*func)(MTYPE,MTYPE, MTYPE, MTYPE), MTYPE scalar1, MTYPE scalar2, Matrix& target) const;
    void _applyLoopScalar(const MTYPE scalar, MTYPE(*func)(MTYPE, MTYPE), Matrix& target) const;
    void _checkBounds(int64 startRow, int64 endRow, int64 startCol, int64 endCol) const;
    void _divideByVector(const Matrix& vec, Matrix& target);
    inline int64 _getNumColsBackEnd() const {
        return _trans == CblasNoTrans ? _numCols : _numRows;
    }
public:
    enum FUNCTION {
        TANH, RECIPROCAL, SQUARE, ABS, EXP, LOG, ZERO, ONE, LOGISTIC1, LOGISTIC2, SIGN
    };
    Matrix();
    Matrix(int64 numRows, int64 numCols);
    Matrix(int64 numRows, int64 numCols, bool transpose);
#ifdef NUMPY_INTERFACE
    Matrix(const PyArrayObject *src);
#endif
    Matrix(const Matrix &like);
    Matrix(MTYPE* data, int64 numRows, int64 numCols);
    Matrix(MTYPE* data, int64 numRows, int64 numCols, bool transpose);
    ~Matrix();

    inline MTYPE& getCell(int64 i, int64 j) const {
        assert(i >= 0 && i < _numRows);
        assert(j >= 0 && j < _numCols);
        if (_trans == CblasTrans) {
            return _data[j * _numRows + i];
        }
        return _data[i * _numCols + j];
    }

    MTYPE& operator()(int64 i, int64 j) const {
        return getCell(i, j);
    }

    inline MTYPE* getData() const {
        return _data;
    }

    inline bool isView() const {
        return !_ownsData;
    }

    inline int64 getNumRows() const {
        return _numRows;
    }

    inline int64 getNumCols() const {
        return _numCols;
    }

    inline int64 getNumDataBytes() const {
        return _numElements * sizeof(MTYPE);
    }

    inline int64 getNumElements() const {
        return _numElements;
    }

    inline int64 getLeadingDim() const {
        return _trans == CblasTrans ? _numRows : _numCols;
    }

    inline int64 getFollowingDim() const {
        return _trans == CblasTrans ? _numCols : _numRows;
    }

    inline CBLAS_TRANSPOSE getBLASTrans() const {
        return _trans;
    }

    inline bool isSameDims(const Matrix& a) const {
        return a.getNumRows() == getNumRows() && a.getNumCols() == getNumCols();
    }

    inline bool isTrans() const {
        return _trans == CblasTrans;
    }

    /*
     * Only use if you know what you're doing!
     * Does not update any dimensions. Just flips the _trans flag.
     *
     * Use transpose() if you want to get the transpose of this matrix.
     */
    inline void setTrans(bool trans) {
        assert(isTrans() == trans || !isView());
        _trans = trans ? CblasTrans : CblasNoTrans;
    }

    void apply(FUNCTION f);
    void apply(Matrix::FUNCTION f, Matrix& target);
    void subtractFromScalar(MTYPE scalar);
    void subtractFromScalar(MTYPE scalar, Matrix &target) const;
    void biggerThanScalar(MTYPE scalar);
    void smallerThanScalar(MTYPE scalar);
    void equalsScalar(MTYPE scalar);
    void biggerThanScalar(MTYPE scalar, Matrix& target) const;
    void smallerThanScalar(MTYPE scalar, Matrix& target) const;
    void equalsScalar(MTYPE scalar, Matrix& target) const;
    void biggerThan(Matrix& a);
    void biggerThan(Matrix& a, Matrix& target) const;
    void smallerThan(Matrix& a);
    void smallerThan(Matrix& a, Matrix& target) const;
    void minWith(Matrix &a);
    void minWith(Matrix &a, Matrix &target) const;
    void maxWith(Matrix &a);
    void maxWith(Matrix &a, Matrix &target) const;
    void equals(Matrix& a);
    void equals(Matrix& a, Matrix& target) const;
    void notEquals(Matrix& a) ;
    void notEquals(Matrix& a, Matrix& target) const;
    void add(const Matrix &m);
    void add(const Matrix &m, MTYPE scale);
    void add(const Matrix &m, MTYPE scaleThis, MTYPE scaleM);
    void add(const Matrix &m, Matrix& target);
    void add(const Matrix &m, MTYPE scaleM, Matrix &target);
    void add(const Matrix &m, MTYPE scaleThis, MTYPE scaleM, Matrix &target);
    void subtract(const Matrix &m);
    void subtract(const Matrix &m, Matrix& target);
    void subtract(const Matrix &m, MTYPE scale);
    void subtract(const Matrix &m, MTYPE scale, Matrix& target);
    void addVector(const Matrix& vec, MTYPE scale);
    void addVector(const Matrix& vec, MTYPE scale, Matrix& target);
    void addVector(const Matrix& vec);
    void addVector(const Matrix& vec, Matrix& target);
    void addScalar(MTYPE scalar);
    void addScalar(MTYPE scalar, Matrix& target) const;
    void maxWithScalar(MTYPE scalar);
    void maxWithScalar(MTYPE scalar, Matrix &target) const;
    void minWithScalar(MTYPE scalar);
    void minWithScalar(MTYPE scalar, Matrix &target) const;
    void eltWiseMultByVector(const Matrix& vec);
    void eltWiseMultByVector(const Matrix& vec, Matrix& target);
    void eltWiseDivideByVector(const Matrix& vec);
    void eltWiseDivideByVector(const Matrix& vec, Matrix& target);
    void resize(int64 newNumRows, int64 newNumCols);
    void resize(const Matrix& like);
    Matrix& slice(int64 startRow, int64 endRow, int64 startCol, int64 endCol) const;
    void slice(int64 startRow, int64 endRow, int64 startCol, int64 endCol, Matrix &target) const;
    Matrix& sliceRows(int64 startRow, int64 endRow) const;
    void sliceRows(int64 startRow, int64 endRow, Matrix& target) const;
    Matrix& sliceCols(int64 startCol, int64 endCol) const;
    void sliceCols(int64 startCol, int64 endCol, Matrix& target) const;
    void rightMult(const Matrix &b, MTYPE scale);
    void rightMult(const Matrix &b, Matrix &target) const;
    void rightMult(const Matrix &b);
    void rightMult(const Matrix &b, MTYPE scaleAB, Matrix &target) const;
    void addProduct(const Matrix &a, const Matrix &b, MTYPE scaleAB, MTYPE scaleThis);
    void addProduct(const Matrix& a, const Matrix& b);
    void eltWiseMult(const Matrix& a);
    void eltWiseMult(const Matrix& a, Matrix& target) const;
    void eltWiseDivide(const Matrix& a);
    void eltWiseDivide(const Matrix& a, Matrix &target) const;
    Matrix& transpose() const;
    Matrix& transpose(bool hard) const;
    Matrix& tile(int64 timesY, int64 timesX) const;
    void tile(int64 timesY, int64 timesX, Matrix& target) const;
    void copy(Matrix &dest, int64 srcStartRow, int64 srcEndRow, int64 srcStartCol, int64 srcEndCol, int64 destStartRow, int64 destStartCol) const;
    Matrix& copy() const;
    void copy(Matrix& target) const;
    Matrix& sum(int64 axis) const;
    void sum(int64 axis, Matrix &target) const;
    MTYPE sum() const;
    MTYPE max() const;
    Matrix& max(int64 axis) const;
    void max(int64 axis, Matrix& target) const;
    MTYPE min() const;
    Matrix& min(int64 axis) const;
    void min(int64 axis, Matrix& target) const;
    MTYPE norm() const;
    MTYPE norm2() const;
    void scale(MTYPE scale);
    void scale(MTYPE alpha, Matrix& target);
    void reshape(int64 numRows, int64 numCols);
    Matrix& reshaped(int64 numRows, int64 numCols);
    void printShape(const char* name) const;
    bool hasNan() const;
    bool hasInf() const;

    void randomizeNormal(MTYPE mean, MTYPE stdev);
    void randomizeUniform();
    void randomizeNormal();
    void print() const;
    void print(int64 startRow,int64 rows, int64 startCol,int64 cols) const;
    void print(int64 rows, int64 cols) const;
};

typedef std::vector<Matrix*> MatrixV;

#endif /* MATRIX_H_ */
