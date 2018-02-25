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

#include "../include/matrix.h"
#include "../include/matrix_funcs.h"

#if defined(_WIN64) || defined(_WIN32)
double sqrt(int _X) {return sqrt((double) _X);}
double log(int _X) {return log((double) _X);}
#endif

using namespace std;

void Matrix::_init(MTYPE* data, int64 numRows, int64 numCols, bool transpose, bool ownsData) {
    _updateDims(numRows, numCols);
    _ownsData = ownsData;
    _trans = transpose ? CblasTrans : CblasNoTrans;
    _data = data;
}

Matrix::Matrix() {
    _init(NULL, 0, 0, false, true);
}

Matrix::Matrix(int64 numRows, int64 numCols) {
    _init(NULL, numRows, numCols, false, true);
    this->_data = numRows * numCols > 0 ? new MTYPE[this->_numElements] : NULL;
}

Matrix::Matrix(int64 numRows, int64 numCols, bool transpose) {
    _init(NULL, numRows, numCols, transpose, true);
    this->_data = numRows * numCols > 0 ? new MTYPE[this->_numElements] : NULL;
}

Matrix::Matrix(const Matrix &like) {
    _init(NULL, like.getNumRows(), like.getNumCols(), false, true);
    this->_data = new MTYPE[this->_numElements];
}

/* construct a matrix with another matrix's data. the resultant
 * matrix does NOT own its data */
Matrix::Matrix(MTYPE* data, int64 numRows, int64 numCols) {
    _init(data, numRows, numCols, false, false);
}

/* construct a matrix with another matrix's data (and optionally transpose it). the resultant
 * matrix does NOT own its data -- it is a VIEW */
Matrix::Matrix(MTYPE* data, int64 numRows, int64 numCols, bool transpose) {
    _init(data, numRows, numCols, transpose, false);
}

#ifdef NUMPY_INTERFACE
Matrix::Matrix(const PyArrayObject *src) {
    this->_data = NULL;
    this->_trans = CblasNoTrans;
    if (src != NULL) {
        this->_updateDims(PyArray_DIM(src,0), PyArray_DIM(src,1));
        if (src->flags & NPY_CONTIGUOUS || src->flags & NPY_FORTRAN) {
            this->_data = (MTYPE*) src->data;
            this->_ownsData = false;
            this->_trans = src->flags & NPY_CONTIGUOUS ? CblasNoTrans : CblasTrans;
        } else {
            this->_data = new MTYPE[PyArray_DIM(src,0) * PyArray_DIM(src,1)];
            for (int64 i = 0; i < PyArray_DIM(src,0); i++) {
                for (int64 j = 0; j < PyArray_DIM(src,1); j++) {
                    (*this)(i,j) = *reinterpret_cast<MTYPE*>(PyArray_GETPTR2(src,i,j));
                }
            }
            this->_ownsData = true;
        }
    }
}
#endif
Matrix::~Matrix() {
    if(this->_data != NULL && this->_ownsData) {
        delete[] this->_data;
    }
}

void Matrix::_updateDims(int64 numRows, int64 numCols) {
    this->_numRows = numRows;
    this->_numCols = numCols;
    this->_numElements = numRows * numCols;
}

void Matrix::_checkBounds(int64 startRow, int64 endRow, int64 startCol, int64 endCol) const {
    assert(startRow >= 0 && startRow <= _numRows);
    assert(endRow >= 0 && endRow <= _numRows);
    assert(startCol >= 0 && startCol <= _numCols);
    assert(endCol >= 0 && endCol <= _numCols);
}

/* will return a view if possible */
Matrix& Matrix::slice(int64 startRow, int64 endRow, int64 startCol, int64 endCol) const {
    endRow = endRow < 0 ? this->_numRows : endRow;
    endCol = endCol < 0 ? this->_numCols : endCol;
    _checkBounds(startRow, endRow, startCol, endCol);
    if (!isTrans() && ((startCol == 0 && endCol == this->_numCols) || (startRow == endRow - 1))) {
        return *new Matrix(this->_data + startRow * this->_numCols + startCol, endRow - startRow, endCol - startCol);
    } else if (isTrans() && ((startRow == 0 && endRow == this->_numRows) || (startCol == endCol - 1))) {
        return *new Matrix(this->_data + startCol * this->_numRows + startRow, endRow - startRow, endCol - startCol, true);
    }
    Matrix& newSlice = *new Matrix(endRow - startRow, endCol - startCol);
    this->copy(newSlice, startRow, endRow, startCol, endCol, 0, 0);
    return newSlice;
}

/* this will NEVER return a view, unlike Matrix_slice */
void Matrix::slice(int64 startRow, int64 endRow, int64 startCol, int64 endCol, Matrix& target) const {
    endRow = endRow < 0 ? this->_numRows : endRow;
    endCol = endCol < 0 ? this->_numCols : endCol;
    _checkBounds(startRow, endRow, startCol, endCol);
    target.resize(endRow - startRow, endCol - startCol);
    this->copy(target, startRow, endRow, startCol, endCol, 0, 0);
}

Matrix& Matrix::sliceRows(int64 startRow, int64 endRow) const {
    return slice(startRow, endRow, 0, -1);
}

void Matrix::sliceRows(int64 startRow, int64 endRow, Matrix& target) const {
    slice(startRow, endRow, 0, -1, target);
}

Matrix& Matrix::sliceCols(int64 startCol, int64 endCol) const {
    return slice(0, -1, startCol, endCol);
}

void Matrix::sliceCols(int64 startCol, int64 endCol, Matrix& target) const {
    slice(0, -1, startCol, endCol, target);
}

void Matrix::subtractFromScalar(MTYPE scalar) {
    subtractFromScalar(scalar, *this);
}

void Matrix::subtractFromScalar(MTYPE scalar, Matrix& target) const {
    if(&target != this) {
        copy(target);
    }
    target.scale(-1);
    target.addScalar(scalar);
}

void Matrix::biggerThanScalar(MTYPE scalar) {
    biggerThanScalar(scalar, *this);
}

void Matrix::smallerThanScalar(MTYPE scalar) {
    smallerThanScalar(scalar, *this);
}

void Matrix::equalsScalar(MTYPE scalar) {
    equalsScalar(scalar, *this);
}

void Matrix::biggerThanScalar(MTYPE scalar, Matrix& target) const {
    target.resize(*this);
    _applyLoopScalar(scalar, &_bigger, target);
}

void Matrix::smallerThanScalar(MTYPE scalar, Matrix& target) const {
    target.resize(*this);
    _applyLoopScalar(scalar, &_smaller, target);
}

void Matrix::equalsScalar(MTYPE scalar, Matrix& target) const {
    target.resize(*this);
    _applyLoopScalar(scalar, &_equal, target);
}

void Matrix::add(const Matrix &m) {
    add(m, 1, *this);
}

void Matrix::add(const Matrix &m, Matrix& target) {
    add(m, 1, target);
}

void Matrix::add(const Matrix &m, MTYPE scale) {
    add(m, scale, *this);
}

void Matrix::subtract(const Matrix &m) {
    add(m, -1, *this);
}

void Matrix::subtract(const Matrix &m, Matrix& target) {
    add(m, -1, target);
}

void Matrix::subtract(const Matrix &m, MTYPE scale) {
    add(m, -scale, *this);
}

void Matrix::subtract(const Matrix &m, MTYPE scale, Matrix& target) {
    add(m, -scale, target);
}

void Matrix::add(const Matrix &m, MTYPE scaleM, Matrix &target) {
    add(m, 1, scaleM, target);
}

void Matrix::add(const Matrix &m, MTYPE scaleThis, MTYPE scaleM) {
    add(m, scaleThis, scaleM, *this);
}

void Matrix::add(const Matrix &m, MTYPE scaleThis, MTYPE scaleM, Matrix &target) {
    assert(this->isSameDims(m));
    if (isTrans() != m.isTrans() || isTrans() != target.isTrans() || scaleThis != 1) {
        if (&target != this) {
            target.resize(*this);
        }
        if(scaleThis == 1 && scaleM == 1) {
            this->_applyLoop2(m, &_add, target);
        } else if (scaleThis == 1) {
            this->_applyLoop2(m, &_addWithScale, scaleM, target);
        } else {
            this->_applyLoop2(m, &_addWithScale2, scaleThis, scaleM, target);
        }
    } else {
        if (&target != this) {
            copy(target);
        }
        CBLAS_AXPY(getNumElements(), scaleM, m._data, 1, target._data, 1);
    }
}

void Matrix::addScalar(MTYPE scalar) {
    addScalar(scalar, *this);
}

void Matrix::addScalar(MTYPE scalar, Matrix& target) const {
    target.resize(*this);
    _applyLoopScalar(scalar, &_add, target);
}

void Matrix::maxWithScalar(MTYPE scalar) {
    maxWithScalar(scalar, *this);
}

void Matrix::maxWithScalar(MTYPE scalar, Matrix& target) const {
    target.resize(*this);
    _applyLoopScalar(scalar, &_max, target);
}

void Matrix::minWithScalar(MTYPE scalar) {
    minWithScalar(scalar, *this);
}

void Matrix::minWithScalar(MTYPE scalar, Matrix& target) const {
    target.resize(*this);
    _applyLoopScalar(scalar, &_min, target);
}

void Matrix::biggerThan(Matrix& a) {
    biggerThan(a, *this);
}

void Matrix::biggerThan(Matrix& a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    _applyLoop2(a, &_bigger, target);
}

void Matrix::smallerThan(Matrix& a) {
    smallerThan(a, *this);
}

void Matrix::smallerThan(Matrix& a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    _applyLoop2(a, &_smaller, target);
}

void Matrix::equals(Matrix& a) {
    equals(a, *this);
}

void Matrix::equals(Matrix& a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    _applyLoop2(a, &_equal, target);
}

void Matrix::notEquals(Matrix& a) {
    notEquals(a, *this);
}

void Matrix::notEquals(Matrix& a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    _applyLoop2(a, &_notEqual, target);
}

void Matrix::minWith(Matrix &a) {
    minWith(a, *this);
}

void Matrix::minWith(Matrix &a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    _applyLoop2(a, &_min, target);
}

void Matrix::maxWith(Matrix &a) {
    maxWith(a, *this);
}

void Matrix::maxWith(Matrix &a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    _applyLoop2(a, &_max, target);
}

/* this := this + scale*tile(vec) */
void Matrix::addVector(const Matrix& vec, MTYPE scale, Matrix& target) {
    if(&target != this) {
        copy(target);
    }
    assert(std::min(vec.getNumCols(), vec.getNumRows()) == 1);
    const bool rowVector = vec.getNumRows() == 1;
    const bool colVector = vec.getNumCols() == 1;
    assert((rowVector && vec.getNumCols() == target.getNumCols()) || (colVector && vec.getNumRows() == target.getNumRows()));
    if (rowVector && colVector) {
        addScalar(vec(0,0) * scale, target);
        return;
    }
    const int64 loopTil = rowVector ? target.getNumRows() : target.getNumCols();
    const int64 dataInc = ((rowVector && target.isTrans()) || (!rowVector && !target.isTrans())) ? 1 : (rowVector ? target.getNumCols() : target.getNumRows());
    const int64 myStride = ((target.isTrans() && rowVector) || (!target.isTrans() && !rowVector)) ? loopTil : 1;
    for (int64 i = 0; i < loopTil; i++) {
        CBLAS_AXPY(vec.getNumElements(), scale, vec._data, 1, target._data + dataInc * i, myStride);
    }
}

/* this := this + scale*tile(vec) */
void Matrix::addVector(const Matrix& vec, MTYPE scale) {
    addVector(vec, scale, *this);
}

void Matrix::addVector(const Matrix& vec) {
    addVector(vec, 1, *this);
}

void Matrix::addVector(const Matrix& vec, Matrix& target) {
    addVector(vec, 1, target);
}

void Matrix::eltWiseMultByVector(const Matrix& vec) {
    eltWiseMultByVector(vec, *this);
}

/* omg test these */
void Matrix::eltWiseMultByVector(const Matrix& vec, Matrix& target) {
    if(&target != this) {
        copy(target);
    }
    assert(std::min(vec.getNumCols(), vec.getNumRows()) == 1);
    const bool rowVector = vec.getNumRows() == 1;
    assert((rowVector && vec.getNumCols() == target.getNumCols()) || (!rowVector && vec.getNumRows() == target.getNumRows()));
    const int64 dataInc = ((rowVector && !target.isTrans()) || (!rowVector && target.isTrans())) ? 1 : (rowVector ? target.getNumRows() : target.getNumCols());
    const int64 myStride = ((!target.isTrans() && !rowVector) || (target.isTrans() && rowVector)) ? 1 : vec.getNumElements();
    const int64 numScaling = rowVector ? target.getNumRows() : target.getNumCols();
    for (int64 i = 0; i < vec.getNumElements(); i++) {
        CBLAS_SCAL(numScaling, vec._data[i], target._data + dataInc * i, myStride);
    }
}

/* return := scale * this * b */
void Matrix::rightMult(const Matrix& b, MTYPE scale) {
    rightMult(b, scale, *this);
}

/* return := this * b */
void Matrix::rightMult(const Matrix& b) {
    rightMult(b, 1);
}

/* target := this * b
 * also resizes target if necessary.*/
void Matrix::rightMult(const Matrix &b, Matrix &target) const {
    rightMult(b, 1, target);
}

/* target := scaleAB * this * b
 * also resizes target if necessary.*/
void Matrix::rightMult(const Matrix &b, MTYPE scaleAB, Matrix &target) const {
    if(&target != this) {
        target.resize(this->_numRows, b._numCols);
    }
    target.addProduct(*this, b, scaleAB, 0);
}

/* this := scaleAB * a*b + scaleC * this
 * ALL SIZES MUST BE CORRECT. */
void Matrix::addProduct(const Matrix& a, const Matrix& b, MTYPE scaleAB, MTYPE scaleThis) {
    assert(a.getNumCols() == b.getNumRows());
    assert(this->getNumRows() == a.getNumRows() && this->getNumCols() == b.getNumCols());
    assert(!isTrans());
    CBLAS_GEMM(CblasRowMajor, a._trans, b._trans, a._numRows, b._numCols, a._numCols, scaleAB, a._data,
            a._getNumColsBackEnd(), b._data, b._getNumColsBackEnd(), scaleThis, this->_data, this->_numCols);
}

void Matrix::addProduct(const Matrix& a, const Matrix& b) {
    addProduct(a, b, 1, 1);
}

Matrix& Matrix::transpose() const {
    return *new Matrix(this->_data, this->_numCols, this->_numRows, !isTrans());
}

Matrix& Matrix::transpose(bool hard) const {
    if (!hard || isTrans()) {
        return transpose();
    }
    Matrix &meTrans = *new Matrix(_numCols, _numRows);
    for (int64 i = 0; i < _numRows; i++) {
        for (int64 j = 0; j < _numCols; j++) {
            meTrans(j, i) = (*this)(i, j);
        }
    }
    return meTrans;
}

Matrix& Matrix::tile(int64 timesY, int64 timesX) const {
    Matrix& tiled = *new Matrix(this->_numRows * timesY, this->_numCols * timesX);
    _tileTo2(tiled);
    return tiled;
}

/* resizes target if necessary */
void Matrix::tile(int64 timesY, int64 timesX, Matrix& target) const {
    target.resize(this->_numRows * timesY, this->_numCols * timesX);
    _tileTo2(target);
}

/* a variant ... seems to be no faster than original. */
void Matrix::_tileTo2(Matrix& target) const {
    for(int64 y = 0; y < target._numRows; y += this->_numRows) {
        for(int64 x = 0; x < target._numCols; x += this->_numCols) {
            this->copy(target, 0, -1, 0, -1, y, x);
        }
    }
}

/* guarantees that result will be non-transposed */
void Matrix::resize(int64 newNumRows, int64 newNumCols) {
    if(this->_numRows != newNumRows || this->_numCols != newNumCols) {
        assert(!isView());
        if (this->getNumElements() != newNumRows * newNumCols) {
            delete[] this->_data; //deleting NULL is ok, sez c++
            this->_data = new MTYPE[newNumRows * newNumCols];
        }
        this->_updateDims(newNumRows, newNumCols);
        this->_trans = CblasNoTrans;
    }
}

void Matrix::resize(const Matrix& like) {
    resize(like.getNumRows(), like.getNumCols());
}

void Matrix::scale(MTYPE alpha) {
    scale(alpha, *this);
}

void Matrix::scale(MTYPE alpha, Matrix& target) {
    if (&target != this) {
        target.resize(*this);
        copy(target);
    }
    CBLAS_SCAL(getNumElements(), alpha, target._data, 1);
}

/* performs no resizing.
 * Warnings:
 * 1. ALL DIMENSIONS MUST BE CORRECT
 * 2. The source and destination memories better not overlap! */
void Matrix::copy(Matrix& dest, int64 srcStartRow, int64 srcEndRow, int64 srcStartCol, int64 srcEndCol, int64 destStartRow, int64 destStartCol) const {
    srcEndRow = srcEndRow < 0 ? this->_numRows : srcEndRow;
    srcEndCol = srcEndCol < 0 ? this->_numCols : srcEndCol;
    assert(destStartRow >= 0 && destStartCol >= 0); //some range-checking
    assert(srcEndRow <= _numRows && srcEndCol <= _numCols);
    assert(destStartRow + srcEndRow - srcStartRow <= dest.getNumRows());
    assert(destStartCol + srcEndCol - srcStartCol <= dest.getNumCols());
    // I found no evidence that memcpy is actually faster than just
    // copying element-by-element.
    if (!isTrans() && !dest.isTrans()) {
        int64 src_start_idx = this->_numCols * srcStartRow + srcStartCol;
        int64 dest_start_idx = dest._numCols * destStartRow + destStartCol;
        int64 copy_row_width = srcEndCol - srcStartCol;

        for (int64 i = srcStartRow; i < srcEndRow; i++) {
            memcpy(dest._data + dest_start_idx + dest._numCols * (i - srcStartRow),
                    this->_data + src_start_idx + this->_numCols * (i - srcStartRow), sizeof(MTYPE) * copy_row_width);
        }
    } else {
        for (int64 i = srcStartRow; i < srcEndRow; i++) {
            for (int64 j = srcStartCol; j < srcEndCol; j++) {
                dest(i - srcStartRow + destStartRow, j - srcStartCol + destStartCol) = (*this)(i, j);
            }
        }
    }
}

/* preserves everything excluding transposedness.
 * new matrix owns its data */
Matrix& Matrix::copy() const {
    Matrix& copy = *new Matrix(*this);
    this->copy(copy);
    return copy;
}

/* resizes target if necessary */
void Matrix::copy(Matrix& target) const {
    target.resize(this->_numRows, this->_numCols); //target is now non-transposed
    if(this->isTrans() == target.isTrans()) {
        this->_copyAllTo(target);
    } else { //if I'm transposed, make sure that target is non-transposed copy
        this->copy(target, 0, -1, 0, -1, 0, 0);
    }
}

void Matrix::_copyAllTo(Matrix& target) const {
    assert(target.isTrans() == isTrans());
    memcpy((void*) target._data, (void*) this->_data, this->getNumDataBytes());
    target._trans = this->_trans;
}

MTYPE Matrix::min() const {
    return _aggregate(&_min, MTYPE_MAX);
}

Matrix& Matrix::min(int64 axis) const {
    Matrix& target = axis == 0 ? *new Matrix(1, this->_numCols) : *new Matrix(this->_numRows, 1);
    this->min(axis, target);
    return target;
}

void Matrix::min(int64 axis, Matrix& target) const {
    _aggregate(axis, target, &_min, MTYPE_MAX);
}

MTYPE Matrix::max() const {
    return _aggregate(&_max, -MTYPE_MAX);
}

Matrix& Matrix::max(int64 axis) const {
    Matrix& target = axis == 0 ? *new Matrix(1, this->_numCols) : *new Matrix(this->_numRows, 1);
    this->max(axis, target);
    return target;
}

void Matrix::max(int64 axis, Matrix& target) const {
    _aggregate(axis, target, &_max, -MTYPE_MAX);
}

MTYPE Matrix::sum() const {
    return _aggregate(&_add, 0);
}

MTYPE Matrix::norm() const {
    return sqrt(norm2());
}

MTYPE Matrix::norm2() const {
    return _aggregate(&_addSquare, 0);
}

Matrix& Matrix::sum(int64 axis) const {
    Matrix& target = axis == 0 ? *new Matrix(1, this->_numCols) : *new Matrix(this->_numRows, 1);
    this->sum(axis, target);
    return target;
}

void Matrix::sum(int64 axis, Matrix& target) const {
    _aggregate(axis, target, &_add, 0);
}

void Matrix::_aggregate(int64 axis, Matrix& target, MTYPE (*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const {
    if (axis == 0) {
        target.resize(1, this->_numCols);
        for (int64 j = 0; j < this->_numCols; j++) {
            target(0, j) = _aggregateCol(j, agg_func, initialValue);
        }
    } else {
        target.resize(this->_numRows, 1);
        for (int64 i = 0; i < this->_numRows; i++) {
            target(i, 0) = _aggregateRow(i, agg_func, initialValue);
        }
    }
}

MTYPE Matrix::_aggregateRow(int64 row, MTYPE (*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const {
    MTYPE v = initialValue;
    for (int64 j = 0; j < this->_numCols; j++) {
        v = agg_func((*this)(row, j), v);
    }
    return v;
}

MTYPE Matrix::_aggregateCol(int64 col, MTYPE (*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const {
    MTYPE v = initialValue;
    for (int64 i = 0; i < this->_numRows; i++) {
        v = agg_func((*this)(i, col), v);
    }
    return v;
}

MTYPE Matrix::_aggregate(MTYPE (*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const {
    MTYPE v = initialValue;
    MTYPE* ptr = _data;
    for (int64 i = 0; i < getNumElements(); i++, ptr++) {
        v = agg_func(*ptr, v);
    }
    return v;
}

void Matrix::printShape(const char* name) const {
    printf("%s: %lldx%lld\n", name, getNumRows(), getNumCols());
}

void Matrix::print() const {
    print(0,getNumRows(),0, getNumCols());
}

void Matrix::print(int64 rows, int64 cols) const {
    print(0,rows,0, cols);
}

void Matrix::print(int64 startRow, int64 rows, int64 startCol, int64 cols) const {
    for (int64 i = startRow; i < std::min(startRow+rows, this->_numRows); i++) {
        for (int64 j = startCol; j < std::min(startCol+cols, this->_numCols); j++) {
            printf("%.15f ", (*this)(i, j));
        }
        printf("\n");
    }
}

void Matrix::apply(Matrix::FUNCTION f) {
    apply(f, *this);
}


void Matrix::apply(Matrix::FUNCTION f, Matrix& target) {
    MTYPE (*func)(MTYPE);
    if(f == EXP) {
        func = &_exp;
    } else if(f == TANH) {
        func = &_tanh;
    } else if(f == RECIPROCAL) {
        func = &_recip;
    } else if (f == SQUARE) {
        func = &_square;
    } else if(f == LOG) {
        func = &_log;
    } else if(f == ZERO) {
        func = &_zero;
    } else if (f == ONE) {
        func = &_one;
    } else if(f == LOGISTIC1) {
        func = &_sigma1;
    } else if(f == LOGISTIC2) {
        func = &_sigma2;
    } else if (f == ABS) {
        func = &_abs;
    } else if (f == SIGN) {
        func = &_sign;
    } else {
        return;
        //LOG(FATAL) << "Matrix::apply: Unknown function type";
    }
    this->_applyLoop(func, target);
}

void Matrix::eltWiseMult(const Matrix& a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    this->_applyLoop2(a, &_mult, target);
}

void Matrix::eltWiseDivide(const Matrix& a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    this->_applyLoop2(a, &_divide, target);
}

void Matrix::eltWiseMult(const Matrix& a) {
    eltWiseMult(a, *this);
}

void Matrix::eltWiseDivide(const Matrix& a) {
    eltWiseDivide(a, *this);
}

void Matrix::randomizeUniform() {
    this->_applyLoop(&_rand);
}

void Matrix::randomizeNormal() {
    //LOG(FATAL) << "randomizeNormal only implemented on MKL!";
}

void Matrix::randomizeNormal(MTYPE mean, MTYPE stdev) {
    //LOG(FATAL) << "randomizeNormal only implemented on MKL!";
}

void Matrix::eltWiseDivideByVector(const Matrix& vec) {
    eltWiseDivideByVector(vec, *this);
}

/* This function allocates a chunk of memory at most as big as the input vector */
void Matrix::eltWiseDivideByVector(const Matrix& vec, Matrix& target) {
    assert(std::min(vec.getNumCols(), vec.getNumRows()) == 1);
    const bool rowVector = vec.getNumRows() == 1;
    assert((rowVector && vec.getNumCols() == getNumCols()) || (!rowVector && vec.getNumRows() == getNumRows()));
    if(&target != this) {
        target.resize(*this);
    }
    _divideByVector(vec, target);
}

void Matrix::_divideByVector(const Matrix& vec, Matrix& target) {
    Matrix& vecInverse = vec.copy();
    vecInverse.apply(RECIPROCAL);
    eltWiseMultByVector(vecInverse,target);
    delete &vecInverse;
}

void Matrix::reshape(int64 numRows, int64 numCols) {
    assert(_numElements == numRows*numCols);
    _numRows = numRows;
    _numCols = numCols;
}

Matrix& Matrix::reshaped(int64 numRows, int64 numCols) {
    assert(_numElements == numRows*numCols);
    return *new Matrix(_data, numRows, numCols, isTrans());
}

void Matrix::_applyLoop(MTYPE (*func)(MTYPE), Matrix& target) {
    MTYPE *ptr = this->_data, *tgtPtr = target._data;
    for (int64 i = 0; i < getNumElements(); i++, ptr++, tgtPtr++) {
        *tgtPtr = (*func)(*ptr);
    }
}

void Matrix::_applyLoop(MTYPE (*func)(MTYPE)) {
    _applyLoop(func, *this);
}

void Matrix::_applyLoop2(const Matrix& a, MTYPE (*func)(MTYPE,MTYPE), Matrix& target) const {
    for (int64 i = 0; i < getNumRows(); i++) {
        for (int64 j = 0; j < getNumCols(); j++) {
            target(i, j) = (*func)((*this)(i, j), a(i, j));
        }
    }
}

void Matrix::_applyLoop2(const Matrix& a, MTYPE (*func)(MTYPE,MTYPE, MTYPE), MTYPE scalar, Matrix& target) const {
    for (int64 i = 0; i < getNumRows(); i++) {
        for (int64 j = 0; j < getNumCols(); j++) {
            target(i, j) = (*func)((*this)(i, j), a(i, j), scalar);
        }
    }
}

void Matrix::_applyLoop2(const Matrix& a, MTYPE (*func)(MTYPE,MTYPE, MTYPE, MTYPE), MTYPE scalar1, MTYPE scalar2, Matrix& target) const {
    for (int64 i = 0; i < getNumRows(); i++) {
        for (int64 j = 0; j < getNumCols(); j++) {
            target(i, j) = (*func)((*this)(i, j), a(i, j), scalar1, scalar2);
        }
    }
}

void Matrix::_applyLoopScalar(const MTYPE scalar, MTYPE(*func)(MTYPE, MTYPE), Matrix& target) const {
    MTYPE *myPtr = _data;
    MTYPE *targetPtr = target._data;
    for (int64 i = 0; i < getNumElements(); i++, myPtr++, targetPtr++) {
        *targetPtr = (*func)(*myPtr, scalar);
    }
}

bool Matrix::hasNan() const {
    for (int64 r = 0; r < _numRows; r++) {
        for (int64 c = 0; c < _numCols; c++) {
            if (isnan((*this)(r,c))) {
                return true;
            }
        }
    }
    return false;
}

bool Matrix::hasInf() const {
    for (int64 r = 0; r < _numRows; r++) {
        for (int64 c = 0; c < _numCols; c++) {
            if (isinf((*this)(r,c))) {
                return true;
            }
        }
    }
    return false;
}


