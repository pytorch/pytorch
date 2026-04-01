#ifndef SLANG_PRELUDE_CPP_TYPES_CORE_H
#define SLANG_PRELUDE_CPP_TYPES_CORE_H

#ifndef SLANG_PRELUDE_ASSERT
#ifdef SLANG_PRELUDE_ENABLE_ASSERT
#define SLANG_PRELUDE_ASSERT(VALUE) assert(VALUE)
#else
#define SLANG_PRELUDE_ASSERT(VALUE)
#endif
#endif

// Since we are using unsigned arithmatic care is need in this comparison.
// It is *assumed* that sizeInBytes >= elemSize. Which means (sizeInBytes >= elemSize) >= 0
// Which means only a single test is needed

// Asserts for bounds checking.
// It is assumed index/count are unsigned types.
#define SLANG_BOUND_ASSERT(index, count) SLANG_PRELUDE_ASSERT(index < count);
#define SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_PRELUDE_ASSERT(index <= (sizeInBytes - elemSize) && (index & 3) == 0);

// Macros to zero index if an access is out of range
#define SLANG_BOUND_ZERO_INDEX(index, count) index = (index < count) ? index : 0;
#define SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    index = (index <= (sizeInBytes - elemSize)) ? index : 0;

// The 'FIX' macro define how the index is fixed. The default is to do nothing. If
// SLANG_ENABLE_BOUND_ZERO_INDEX the fix macro will zero the index, if out of range
#ifdef SLANG_ENABLE_BOUND_ZERO_INDEX
#define SLANG_BOUND_FIX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#else
#define SLANG_BOUND_FIX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

#ifndef SLANG_BOUND_CHECK
#define SLANG_BOUND_CHECK(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX(index, count)
#endif

#ifndef SLANG_BOUND_CHECK_BYTE_ADDRESS
#define SLANG_BOUND_CHECK_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes)    \
    SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#endif

#ifndef SLANG_BOUND_CHECK_FIXED_ARRAY
#define SLANG_BOUND_CHECK_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

struct TypeInfo
{
    size_t typeSize;
};

template<typename T, size_t SIZE>
struct FixedArray
{
    const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }
    T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }

    T m_data[SIZE];
};

// An array that has no specified size, becomes a 'Array'. This stores the size so it can
// potentially do bounds checking.
template<typename T>
struct Array
{
    const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }

    T* data;
    size_t count;
};

/* Constant buffers become a pointer to the contained type, so ConstantBuffer<T> becomes T* in C++
 * code.
 */

template<typename T, int COUNT>
struct Vector;

template<typename T>
struct Vector<T, 1>
{
    T x;
    const T& operator[](size_t /*index*/) const { return x; }
    T& operator[](size_t /*index*/) { return x; }
    operator T() const { return x; }
    Vector() = default;
    Vector(T scalar) { x = scalar; }
    template<typename U>
    Vector(Vector<U, 1> other)
    {
        x = (T)other.x;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 1;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T>
struct Vector<T, 2>
{
    T x, y;
    const T& operator[](size_t index) const { return index == 0 ? x : y; }
    T& operator[](size_t index) { return index == 0 ? x : y; }
    Vector() = default;
    Vector(T scalar) { x = y = scalar; }
    Vector(T _x, T _y)
    {
        x = _x;
        y = _y;
    }
    template<typename U>
    Vector(Vector<U, 2> other)
    {
        x = (T)other.x;
        y = (T)other.y;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 2;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T>
struct Vector<T, 3>
{
    T x, y, z;
    const T& operator[](size_t index) const { return *((T*)(this) + index); }
    T& operator[](size_t index) { return *((T*)(this) + index); }

    Vector() = default;
    Vector(T scalar) { x = y = z = scalar; }
    Vector(T _x, T _y, T _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }
    template<typename U>
    Vector(Vector<U, 3> other)
    {
        x = (T)other.x;
        y = (T)other.y;
        z = (T)other.z;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 3;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T>
struct Vector<T, 4>
{
    T x, y, z, w;

    const T& operator[](size_t index) const { return *((T*)(this) + index); }
    T& operator[](size_t index) { return *((T*)(this) + index); }
    Vector() = default;
    Vector(T scalar) { x = y = z = w = scalar; }
    Vector(T _x, T _y, T _z, T _w)
    {
        x = _x;
        y = _y;
        z = _z;
        w = _w;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 4;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T, int N>
SLANG_FORCE_INLINE Vector<T, N> _slang_select(
    Vector<bool, N> condition,
    Vector<T, N> v0,
    Vector<T, N> v1)
{
    Vector<T, N> result;
    for (int i = 0; i < N; i++)
    {
        result[i] = condition[i] ? v0[i] : v1[i];
    }
    return result;
}

template<typename T>
SLANG_FORCE_INLINE T _slang_select(bool condition, T v0, T v1)
{
    return condition ? v0 : v1;
}

template<typename T, int N>
SLANG_FORCE_INLINE T _slang_vector_get_element(Vector<T, N> x, int index)
{
    return x[index];
}

template<typename T, int N>
SLANG_FORCE_INLINE const T* _slang_vector_get_element_ptr(const Vector<T, N>* x, int index)
{
    return &((*const_cast<Vector<T, N>*>(x))[index]);
}

template<typename T, int N>
SLANG_FORCE_INLINE T* _slang_vector_get_element_ptr(Vector<T, N>* x, int index)
{
    return &((*x)[index]);
}

template<typename T, int n, typename OtherT, int m>
SLANG_FORCE_INLINE Vector<T, n> _slang_vector_reshape(const Vector<OtherT, m> other)
{
    Vector<T, n> result;
    for (int i = 0; i < n; i++)
    {
        OtherT otherElement = T(0);
        if (i < m)
            otherElement = _slang_vector_get_element(other, i);
        *_slang_vector_get_element_ptr(&result, i) = (T)otherElement;
    }
    return result;
}

typedef uint32_t uint;

#define SLANG_VECTOR_BINARY_OP(T, op)            \
    template<int n>                              \
    SLANG_FORCE_INLINE Vector<T, n> operator op( \
        const Vector<T, n>& thisVal,             \
        const Vector<T, n>& other)               \
    {                                            \
        Vector<T, n> result;                     \
        for (int i = 0; i < n; i++)              \
            result[i] = thisVal[i] op other[i];  \
        return result;                           \
    }
#define SLANG_VECTOR_BINARY_COMPARE_OP(T, op)       \
    template<int n>                                 \
    SLANG_FORCE_INLINE Vector<bool, n> operator op( \
        const Vector<T, n>& thisVal,                \
        const Vector<T, n>& other)                  \
    {                                               \
        Vector<bool, n> result;                     \
        for (int i = 0; i < n; i++)                 \
            result[i] = thisVal[i] op other[i];     \
        return result;                              \
    }

#define SLANG_VECTOR_UNARY_OP(T, op)                                         \
    template<int n>                                                          \
    SLANG_FORCE_INLINE Vector<T, n> operator op(const Vector<T, n>& thisVal) \
    {                                                                        \
        Vector<T, n> result;                                                 \
        for (int i = 0; i < n; i++)                                          \
            result[i] = op thisVal[i];                                       \
        return result;                                                       \
    }
#define SLANG_INT_VECTOR_OPS(T)           \
    SLANG_VECTOR_BINARY_OP(T, +)          \
    SLANG_VECTOR_BINARY_OP(T, -)          \
    SLANG_VECTOR_BINARY_OP(T, *)          \
    SLANG_VECTOR_BINARY_OP(T, /)          \
    SLANG_VECTOR_BINARY_OP(T, &)          \
    SLANG_VECTOR_BINARY_OP(T, |)          \
    SLANG_VECTOR_BINARY_OP(T, &&)         \
    SLANG_VECTOR_BINARY_OP(T, ||)         \
    SLANG_VECTOR_BINARY_OP(T, ^)          \
    SLANG_VECTOR_BINARY_OP(T, %)          \
    SLANG_VECTOR_BINARY_OP(T, >>)         \
    SLANG_VECTOR_BINARY_OP(T, <<)         \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, ==) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, !=) \
    SLANG_VECTOR_UNARY_OP(T, !)           \
    SLANG_VECTOR_UNARY_OP(T, ~)
#define SLANG_FLOAT_VECTOR_OPS(T)         \
    SLANG_VECTOR_BINARY_OP(T, +)          \
    SLANG_VECTOR_BINARY_OP(T, -)          \
    SLANG_VECTOR_BINARY_OP(T, *)          \
    SLANG_VECTOR_BINARY_OP(T, /)          \
    SLANG_VECTOR_UNARY_OP(T, -)           \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, ==) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, !=)

SLANG_INT_VECTOR_OPS(bool)
SLANG_INT_VECTOR_OPS(int)
SLANG_INT_VECTOR_OPS(int8_t)
SLANG_INT_VECTOR_OPS(int16_t)
SLANG_INT_VECTOR_OPS(int64_t)
SLANG_INT_VECTOR_OPS(uint)
SLANG_INT_VECTOR_OPS(uint8_t)
SLANG_INT_VECTOR_OPS(uint16_t)
SLANG_INT_VECTOR_OPS(uint64_t)

SLANG_FLOAT_VECTOR_OPS(float)
SLANG_FLOAT_VECTOR_OPS(double)

#define SLANG_VECTOR_INT_NEG_OP(T)                      \
    template<int N>                                     \
    Vector<T, N> operator-(const Vector<T, N>& thisVal) \
    {                                                   \
        Vector<T, N> result;                            \
        for (int i = 0; i < N; i++)                     \
            result[i] = 0 - thisVal[i];                 \
        return result;                                  \
    }
SLANG_VECTOR_INT_NEG_OP(int)
SLANG_VECTOR_INT_NEG_OP(int8_t)
SLANG_VECTOR_INT_NEG_OP(int16_t)
SLANG_VECTOR_INT_NEG_OP(int64_t)
SLANG_VECTOR_INT_NEG_OP(uint)
SLANG_VECTOR_INT_NEG_OP(uint8_t)
SLANG_VECTOR_INT_NEG_OP(uint16_t)
SLANG_VECTOR_INT_NEG_OP(uint64_t)

#define SLANG_FLOAT_VECTOR_MOD(T)                                               \
    template<int N>                                                             \
    Vector<T, N> operator%(const Vector<T, N>& left, const Vector<T, N>& right) \
    {                                                                           \
        Vector<T, N> result;                                                    \
        for (int i = 0; i < N; i++)                                             \
            result[i] = _slang_fmod(left[i], right[i]);                         \
        return result;                                                          \
    }

SLANG_FLOAT_VECTOR_MOD(float)
SLANG_FLOAT_VECTOR_MOD(double)
#undef SLANG_FLOAT_VECTOR_MOD
#undef SLANG_VECTOR_BINARY_OP
#undef SLANG_VECTOR_UNARY_OP
#undef SLANG_INT_VECTOR_OPS
#undef SLANG_FLOAT_VECTOR_OPS
#undef SLANG_VECTOR_INT_NEG_OP
#undef SLANG_FLOAT_VECTOR_MOD

template<typename T, int ROWS, int COLS>
struct Matrix
{
    Vector<T, COLS> rows[ROWS];
    const Vector<T, COLS>& operator[](size_t index) const { return rows[index]; }
    Vector<T, COLS>& operator[](size_t index) { return rows[index]; }
    Matrix() = default;
    Matrix(T scalar)
    {
        for (int i = 0; i < ROWS; i++)
            rows[i] = Vector<T, COLS>(scalar);
    }
    Matrix(const Vector<T, COLS>& row0) { rows[0] = row0; }
    Matrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1)
    {
        rows[0] = row0;
        rows[1] = row1;
    }
    Matrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1, const Vector<T, COLS>& row2)
    {
        rows[0] = row0;
        rows[1] = row1;
        rows[2] = row2;
    }
    Matrix(
        const Vector<T, COLS>& row0,
        const Vector<T, COLS>& row1,
        const Vector<T, COLS>& row2,
        const Vector<T, COLS>& row3)
    {
        rows[0] = row0;
        rows[1] = row1;
        rows[2] = row2;
        rows[3] = row3;
    }
    template<typename U, int otherRow, int otherCol>
    Matrix(const Matrix<U, otherRow, otherCol>& other)
    {
        int minRow = ROWS;
        int minCol = COLS;
        if (minRow > otherRow)
            minRow = otherRow;
        if (minCol > otherCol)
            minCol = otherCol;
        for (int i = 0; i < minRow; i++)
            for (int j = 0; j < minCol; j++)
                rows[i][j] = (T)other.rows[i][j];
    }
    Matrix(T v0, T v1, T v2, T v3)
    {
        rows[0][0] = v0;
        rows[0][1] = v1;
        rows[1][0] = v2;
        rows[1][1] = v3;
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5)
    {
        if (COLS == 3)
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[1][0] = v3;
            rows[1][1] = v4;
            rows[1][2] = v5;
        }
        else
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[1][0] = v2;
            rows[1][1] = v3;
            rows[2][0] = v4;
            rows[2][1] = v5;
        }
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7)
    {
        if (COLS == 4)
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[0][3] = v3;
            rows[1][0] = v4;
            rows[1][1] = v5;
            rows[1][2] = v6;
            rows[1][3] = v7;
        }
        else
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[1][0] = v2;
            rows[1][1] = v3;
            rows[2][0] = v4;
            rows[2][1] = v5;
            rows[3][0] = v6;
            rows[3][1] = v7;
        }
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8)
    {
        rows[0][0] = v0;
        rows[0][1] = v1;
        rows[0][2] = v2;
        rows[1][0] = v3;
        rows[1][1] = v4;
        rows[1][2] = v5;
        rows[2][0] = v6;
        rows[2][1] = v7;
        rows[2][2] = v8;
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11)
    {
        if (COLS == 4)
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[0][3] = v3;
            rows[1][0] = v4;
            rows[1][1] = v5;
            rows[1][2] = v6;
            rows[1][3] = v7;
            rows[2][0] = v8;
            rows[2][1] = v9;
            rows[2][2] = v10;
            rows[2][3] = v11;
        }
        else
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[1][0] = v3;
            rows[1][1] = v4;
            rows[1][2] = v5;
            rows[2][0] = v6;
            rows[2][1] = v7;
            rows[2][2] = v8;
            rows[3][0] = v9;
            rows[3][1] = v10;
            rows[3][2] = v11;
        }
    }
    Matrix(
        T v0,
        T v1,
        T v2,
        T v3,
        T v4,
        T v5,
        T v6,
        T v7,
        T v8,
        T v9,
        T v10,
        T v11,
        T v12,
        T v13,
        T v14,
        T v15)
    {
        rows[0][0] = v0;
        rows[0][1] = v1;
        rows[0][2] = v2;
        rows[0][3] = v3;
        rows[1][0] = v4;
        rows[1][1] = v5;
        rows[1][2] = v6;
        rows[1][3] = v7;
        rows[2][0] = v8;
        rows[2][1] = v9;
        rows[2][2] = v10;
        rows[2][3] = v11;
        rows[3][0] = v12;
        rows[3][1] = v13;
        rows[3][2] = v14;
        rows[3][3] = v15;
    }
};

#define SLANG_MATRIX_BINARY_OP(T, op)                                                         \
    template<int R, int C>                                                                    \
    Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal, const Matrix<T, R, C>& other) \
    {                                                                                         \
        Matrix<T, R, C> result;                                                               \
        for (int i = 0; i < R; i++)                                                           \
            for (int j = 0; j < C; j++)                                                       \
                result.rows[i][j] = thisVal.rows[i][j] op other.rows[i][j];                   \
        return result;                                                                        \
    }

#define SLANG_MATRIX_BINARY_COMPARE_OP(T, op)                                                    \
    template<int R, int C>                                                                       \
    Matrix<bool, R, C> operator op(const Matrix<T, R, C>& thisVal, const Matrix<T, R, C>& other) \
    {                                                                                            \
        Matrix<bool, R, C> result;                                                               \
        for (int i = 0; i < R; i++)                                                              \
            for (int j = 0; j < C; j++)                                                          \
                result.rows[i][j] = thisVal.rows[i][j] op other.rows[i][j];                      \
        return result;                                                                           \
    }

#define SLANG_MATRIX_UNARY_OP(T, op)                            \
    template<int R, int C>                                      \
    Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal) \
    {                                                           \
        Matrix<T, R, C> result;                                 \
        for (int i = 0; i < R; i++)                             \
            for (int j = 0; j < C; j++)                         \
                result[i].rows[i][j] = op thisVal.rows[i][j];   \
        return result;                                          \
    }

#define SLANG_INT_MATRIX_OPS(T)           \
    SLANG_MATRIX_BINARY_OP(T, +)          \
    SLANG_MATRIX_BINARY_OP(T, -)          \
    SLANG_MATRIX_BINARY_OP(T, *)          \
    SLANG_MATRIX_BINARY_OP(T, /)          \
    SLANG_MATRIX_BINARY_OP(T, &)          \
    SLANG_MATRIX_BINARY_OP(T, |)          \
    SLANG_MATRIX_BINARY_OP(T, &&)         \
    SLANG_MATRIX_BINARY_OP(T, ||)         \
    SLANG_MATRIX_BINARY_OP(T, ^)          \
    SLANG_MATRIX_BINARY_OP(T, %)          \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, ==) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, !=) \
    SLANG_MATRIX_UNARY_OP(T, !)           \
    SLANG_MATRIX_UNARY_OP(T, ~)
#define SLANG_FLOAT_MATRIX_OPS(T)         \
    SLANG_MATRIX_BINARY_OP(T, +)          \
    SLANG_MATRIX_BINARY_OP(T, -)          \
    SLANG_MATRIX_BINARY_OP(T, *)          \
    SLANG_MATRIX_BINARY_OP(T, /)          \
    SLANG_MATRIX_UNARY_OP(T, -)           \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, ==) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, !=)
SLANG_INT_MATRIX_OPS(int)
SLANG_INT_MATRIX_OPS(int8_t)
SLANG_INT_MATRIX_OPS(int16_t)
SLANG_INT_MATRIX_OPS(int64_t)
SLANG_INT_MATRIX_OPS(uint)
SLANG_INT_MATRIX_OPS(uint8_t)
SLANG_INT_MATRIX_OPS(uint16_t)
SLANG_INT_MATRIX_OPS(uint64_t)

SLANG_FLOAT_MATRIX_OPS(float)
SLANG_FLOAT_MATRIX_OPS(double)

#define SLANG_MATRIX_INT_NEG_OP(T)                                        \
    template<int R, int C>                                                \
    SLANG_FORCE_INLINE Matrix<T, R, C> operator-(Matrix<T, R, C> thisVal) \
    {                                                                     \
        Matrix<T, R, C> result;                                           \
        for (int i = 0; i < R; i++)                                       \
            for (int j = 0; j < C; j++)                                   \
                result.rows[i][j] = 0 - thisVal.rows[i][j];               \
        return result;                                                    \
    }
SLANG_MATRIX_INT_NEG_OP(int)
SLANG_MATRIX_INT_NEG_OP(int8_t)
SLANG_MATRIX_INT_NEG_OP(int16_t)
SLANG_MATRIX_INT_NEG_OP(int64_t)
SLANG_MATRIX_INT_NEG_OP(uint)
SLANG_MATRIX_INT_NEG_OP(uint8_t)
SLANG_MATRIX_INT_NEG_OP(uint16_t)
SLANG_MATRIX_INT_NEG_OP(uint64_t)

#define SLANG_FLOAT_MATRIX_MOD(T)                                                             \
    template<int R, int C>                                                                    \
    SLANG_FORCE_INLINE Matrix<T, R, C> operator%(Matrix<T, R, C> left, Matrix<T, R, C> right) \
    {                                                                                         \
        Matrix<T, R, C> result;                                                               \
        for (int i = 0; i < R; i++)                                                           \
            for (int j = 0; j < C; j++)                                                       \
                result.rows[i][j] = _slang_fmod(left.rows[i][j], right.rows[i][j]);           \
        return result;                                                                        \
    }

SLANG_FLOAT_MATRIX_MOD(float)
SLANG_FLOAT_MATRIX_MOD(double)
#undef SLANG_FLOAT_MATRIX_MOD
#undef SLANG_MATRIX_BINARY_OP
#undef SLANG_MATRIX_UNARY_OP
#undef SLANG_INT_MATRIX_OPS
#undef SLANG_FLOAT_MATRIX_OPS
#undef SLANG_MATRIX_INT_NEG_OP
#undef SLANG_FLOAT_MATRIX_MOD

template<typename TResult, typename TInput>
TResult slang_bit_cast(TInput val)
{
    return *(TResult*)(&val);
}

#endif
