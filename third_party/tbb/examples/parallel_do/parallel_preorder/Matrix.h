/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

class Matrix {
    static const int n = 20;
    float array[n][n];
public:
    Matrix() {}
    Matrix( float z ) {
        for( int i=0; i<n; ++i )
            for( int j=0; j<n; ++j )
                array[i][j] = i==j ? z : 0;
    }
    friend Matrix operator-( const Matrix& x ) {
        Matrix result;
        for( int i=0; i<n; ++i )
            for( int j=0; j<n; ++j )
                result.array[i][j] = -x.array[i][j];
        return result;
    }
    friend Matrix operator+( const Matrix& x, const Matrix& y ) {
        Matrix result;
        for( int i=0; i<n; ++i )
            for( int j=0; j<n; ++j )
                result.array[i][j] = x.array[i][j] + y.array[i][j];
        return result;
    }
    friend Matrix operator-( const Matrix& x, const Matrix& y ) {
        Matrix result;
        for( int i=0; i<n; ++i )
            for( int j=0; j<n; ++j )
                result.array[i][j] = x.array[i][j] - y.array[i][j];
        return result;
    }
    friend Matrix operator*( const Matrix& x, const Matrix& y ) {
        Matrix result(0);
        for( int i=0; i<n; ++i ) 
            for( int k=0; k<n; ++k )
                for( int j=0; j<n; ++j )
                    result.array[i][j] += x.array[i][k] * y.array[k][j];
        return result;
    }
};
