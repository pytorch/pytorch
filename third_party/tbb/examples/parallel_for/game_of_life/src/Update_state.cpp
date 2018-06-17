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

#include "Evolution.h"

#ifdef USE_SSE 
/* Update states with SSE */

#include <xmmintrin.h>
#include <emmintrin.h>

inline void create_record( char * src, unsigned * dst, unsigned width)
{
    dst[0] |= src[width - 1];
    for( unsigned a=0; a<31u; ++a )
        dst[0] |= src[a]<<(a+1);
    unsigned a;
    for( a=31u; a<width; ++a )
        dst[(a+1)/32u] |= src[a]<<((a+1)%32u);
    dst[(a+1)/32u] |= src[0]<<((a+1)%32u);
}

inline void sum_offset( __m128i * X, __m128i * A, __m128i * B, __m128i * C, 
                        unsigned size_sse_ar, unsigned shift )
{
    for(unsigned i=0; i<size_sse_ar; ++i) 
    {
        __m128i tmp = _mm_and_si128(A[i],X[shift + i]);    
        A[i]=_mm_xor_si128(A[i],X[shift + i]);    
        C[i]=_mm_or_si128(C[i],_mm_and_si128(B[i],tmp));
        B[i]=_mm_xor_si128(B[i],tmp);
    }
}

inline void shift_left2D( __m128i * X, unsigned height, unsigned size_sse_row )
{
    for( unsigned b=0; b<height; ++b ) 
    {
        unsigned ind = b*size_sse_row;
        unsigned x0 = X[ind].m128i_u32[0] & 1;

        X[ind] =_mm_or_si128( _mm_srli_epi16(X[ind],1), 
            _mm_slli_epi16( _mm_srli_si128( X[ind], 2), 15) );
    
        unsigned x1 = X[ind + 1].m128i_u32[0] & 1;
        X[ind+1] =_mm_or_si128( _mm_srli_epi16( X[ind+1],1), 
            _mm_slli_epi16( _mm_srli_si128( X[ind+1], 2), 15) );
        X[ind].m128i_u32[3] |= x1<<31;
        
        unsigned x2 = X[ind + 2].m128i_u32[0] & 1;
        X[ind+2] =_mm_or_si128( _mm_srli_epi16( X[ind+2],1), 
            _mm_slli_epi16( _mm_srli_si128( X[ind+2], 2), 15) );
        X[ind+1].m128i_u32[3] |= x2<<31;
        
        unsigned* dst = (unsigned*)&X[ind];
        dst[301/32u] |= x0<<(301%32u);
   }
}

inline void shift_right2D( __m128i * X, unsigned height, unsigned size_sse_row )
{
    for( unsigned b=0; b<height; ++b ) 
    {
        unsigned ind = b*size_sse_row;

        unsigned x0 = X[ind].m128i_u32[3]; x0>>=31;
        X[ind] =_mm_or_si128( _mm_slli_epi16(X[ind],1), 
            _mm_srli_epi16( _mm_slli_si128( X[ind], 2), 15) );
            
        unsigned x1 = X[ind + 1].m128i_u32[3]; x1>>=31;
        X[ind + 1] =_mm_or_si128( _mm_slli_epi16(X[ind + 1],1),
                _mm_srli_epi16( _mm_slli_si128( X[ind + 1], 2), 15) );
        X[ind + 1].m128i_u32[0] |= x0;
                
        unsigned* dst = (unsigned*)&X[ind];
        unsigned x2 = dst[301/32u] & (1<<(301%32u)); x2>>=(301%32u);
        X[ind + 2] =_mm_or_si128( _mm_slli_epi16(X[ind + 2],1),
            _mm_srli_epi16( _mm_slli_si128( X[ind + 2], 2), 15) );        
        X[ind + 2].m128i_u32[0] |= x1;    
        X[ind].m128i_u32[0] |= x2;
   }
}

void UpdateState(Matrix * m_matrix, char * dest ,int begin, int end)
{
    //300/128 + 1 =3, 3*300=900
    unsigned size_sse_row = m_matrix->width/128 + 1; //3
    unsigned size_sse_ar=size_sse_row * (end - begin); 
    __m128i X[906], A[900], B[900], C[900];
    char * mas  = m_matrix->data;
    
    for( unsigned i=0; i<size_sse_ar; ++i)
    {
        A[i].m128i_u32[0]=0;A[i].m128i_u32[1]=0;A[i].m128i_u32[2]=0;A[i].m128i_u32[3]=0;
        B[i].m128i_u32[0]=0;B[i].m128i_u32[1]=0;B[i].m128i_u32[2]=0;B[i].m128i_u32[3]=0;
        C[i].m128i_u32[0]=0;C[i].m128i_u32[1]=0;C[i].m128i_u32[2]=0;C[i].m128i_u32[3]=0;    
    }

    for( unsigned i=0; i<size_sse_ar+6; ++i)
    {
        X[i].m128i_u32[0]=0;X[i].m128i_u32[1]=0;X[i].m128i_u32[2]=0;X[i].m128i_u32[3]=0;
    }

    // create X[] with bounds
    unsigned height = end - begin;
    unsigned width = m_matrix->width;
    for( unsigned b = 0 ; b < height; ++b ) 
    {
        char* src = &mas[(b + begin)*width];
        unsigned* dst = (unsigned*)&X[(b+1)*size_sse_row];
        create_record(src, dst, width);
    }
    // create high row in X[]
    char * src;
    if(begin == 0) 
    {
        src = &mas[(m_matrix->height-1)*width];
    }
    else 
    {
        src = &mas[(begin-1)*width];
    }
    unsigned* dst = (unsigned*)X;
    create_record(src, dst, width);
    
    //create lower row in X[]
    if(end == m_matrix->height ) 
    {
        src = mas;
    }        
    else 
    {
        src = &mas[end*width];
    }
    dst = (unsigned*)&X[(height+1)*size_sse_row];
    create_record(src, dst, width);
    
    //sum( C, B, A, X+offset_for_upwards ); high-left friend
    sum_offset(X,A,B,C,size_sse_ar, 0);
    
    //sum( C, B, A, X+offset_for_no_vertical_shift );
    sum_offset(X,A,B,C,size_sse_ar, size_sse_row);
    
    //sum( C, B, A, X+offset_for_downwards );
    sum_offset(X,A,B,C,size_sse_ar, 2*size_sse_row);

    //shift_left( X ); (when view 2D) in our logic it is in right
    height = end - begin + 2;
    shift_left2D( X, height, size_sse_row);

    //sum( C, B, A, X+offset_for_upwards ); high-left friend
    sum_offset(X,A,B,C,size_sse_ar, 0);

    //sum( C, B, A, X+offset_for_downwards );
    sum_offset(X,A,B,C,size_sse_ar, 2*size_sse_row);

    //shift_left( X ); (view in 2D) in our logic it is right shift
    height = end - begin + 2;
    shift_left2D( X, height, size_sse_row);
    
    //sum( C, B, A, X+offset_for_upwards ); high-right friend
    sum_offset(X,A,B,C,size_sse_ar, 0);
    
    //sum( C, B, A, X+offset_for_no_vertical_shift ); right friend
    sum_offset(X,A,B,C,size_sse_ar, size_sse_row);    
    
    //sum( C, B, A, X+offset_for_downwards ); right down friend
    sum_offset(X,A,B,C,size_sse_ar, 2*size_sse_row);

    //shift_right( X ); (when view in 2D) in our case it left shift.
    height = end - begin + 2;
    shift_right2D( X, height, size_sse_row);
    
    //X = (X|A)&B&~C (done bitwise over the arrays) 
    unsigned shift = size_sse_row;
    for(unsigned i=0; i<size_sse_ar; ++i) 
    {
        C[i].m128i_u32[0] = ~C[i].m128i_u32[0];
        C[i].m128i_u32[1] = ~C[i].m128i_u32[1];
        C[i].m128i_u32[2] = ~C[i].m128i_u32[2];
        C[i].m128i_u32[3] = ~C[i].m128i_u32[3];
        X[shift + i] = _mm_and_si128(_mm_and_si128(_mm_or_si128(X[shift + i],
            A[i]),B[i]),C[i]);    
    }

    height = end - begin;
    width=m_matrix->width;
    for( unsigned b=0; b<height; ++b ) 
    {
        char* dst = &dest[(b+begin)*width];
        unsigned* src = (unsigned*)&X[(b+1)*size_sse_row];
        for( unsigned a=0; a<width; ++a )
        {
            unsigned c = src[a/32u] & 1<<(a%32u);
            dst[a] = c>>(a%32u);
        }
    }
}
#else 
/* end SSE block */

// ----------------------------------------------------------------------
// GetAdjacentCellState() - returns the state (value) of the specified 
// adjacent cell of the current cell "cellNumber"
char GetAdjacentCellState(
                                char* source,      // pointer to source data block
                                int x,             // logical width of field
                                int y,             // logical height of field
                                int cellNumber,    // number of cell position to examine
                                int cp             // which adjacent position
                               )
{
/* 
cp 
*-- cp=1 ... --- cp=8 (summary: -1-2-3-
-x-          -x-                -4-x-5-
---          --*                -6-7-8- )
*/
    char cellState = 0;        // return value

    // set up boundary flags to trigger field-wrap logic
    bool onTopRow = false;
    bool onBottomRow = false;
    bool onLeftColumn = false;
    bool onRightColumn = false;

    // check to see if cell is on top row
    if (cellNumber < x)
    {
        onTopRow = true;
    }
    // check to see if cell is on bottom row
    if ((x*y)-cellNumber <= x)
    {
        onBottomRow = true;
    }
    // check to see if cell is on left column
    if (cellNumber%x == 0)
    {
        onLeftColumn = true;
    }
    // check to see if cell is on right column
    if ((cellNumber+1)%x == 0)
    {
        onRightColumn = true;
    }

    switch (cp)
    {
        case 1:
            if (onTopRow && onLeftColumn)
            {
                return *(source+((x*y)-1));
            }
            if (onTopRow && !onLeftColumn)
            {
                return *(source+(((x*y)-x)+(cellNumber-1)));
            }
            if (onLeftColumn && !onTopRow)
            {
                return *(source+(cellNumber-1));
            }
            return *((source+cellNumber)-(x+1));

        case 2:
            if (onTopRow)
            {
                return *(source+(((x*y)-x)+cellNumber));
            }
            return *((source+cellNumber)-x);

        case 3:
            if (onTopRow && onRightColumn)
            {
                return *(source+((x*y)-x));
            }
            if (onTopRow && !onRightColumn)
            {
                return *(source+(((x*y)-x)+(cellNumber+1)));
            }
            if (onRightColumn && !onTopRow)
            {
                return *(source+((cellNumber-(x*2))+1));
            }
            return *(source+(cellNumber-(x-1)));

        case 4:
            if (onRightColumn)
            {
                return *(source+(cellNumber-(x-1)));
            }
            return *(source+(cellNumber+1));

        case 5:
            if (onBottomRow && onRightColumn)
            {
                return *source;
            }
            if (onBottomRow && !onRightColumn)
            {
                return *(source+((cellNumber-((x*y)-x))+1));
            }
            if (onRightColumn && !onBottomRow)
            {
                return *(source+(cellNumber+1));
            }
            return *(source+(((cellNumber+x))+1));

        case 6:
            if (onBottomRow)
            {
                return *(source+(cellNumber-((x*y)-x)));
            }
            return *(source+(cellNumber+x));

        case 7:
            if (onBottomRow && onLeftColumn)
            {
                return *(source+(x-1));
            }
            if (onBottomRow && !onLeftColumn)
            {
                return *(source+(cellNumber-((x*y)-x)-1));
            }
            if (onLeftColumn && !onBottomRow)
            {
                return *(source+(cellNumber+((x*2)-1)));
            }
            return *(source+(cellNumber+(x-1)));

        case 8:
            if (onLeftColumn)
            {
                return *(source+(cellNumber+(x-1)));
            }
            return *(source+(cellNumber-1));
    }
    return cellState;
}

char CheckCell(Matrix * m_matrix, int cellNumber)
{
    char total = 0;
    char* source = m_matrix->data;
    //look around to find cell's with status "alive"
    for(int i=1; i<9; i++)
    {
        total += GetAdjacentCellState(source, m_matrix->width, m_matrix->height, cellNumber, i);
    }
    // if the number of adjacent live cells is < 2 or > 3, the result is a dead 
    // cell regardless of its current state. (A live cell dies of loneliness if it
    // has less than 2 neighbors, and of overcrowding if it has more than 3; a new
    // cell is born in an empty spot only if it has exactly 3 neighbors.
    if (total < 2 || total > 3)
    {
        return 0;
    }

    // if we get here and the cell position holds a living cell, it stays alive
    if (*(source+cellNumber))
    {
        return 1;
    }

    // we have an empty position. If there are only 2 neighbors, the position stays
    // empty.
    if (total == 2)
    {
        return 0;
    }

    // we have an empty position and exactly 3 neighbors. A cell is born.
    return 1;
}

void UpdateState(Matrix * m_matrix, char * dest ,int begin, int end)
{
        for (int i=begin; i<=end; i++)
        {
            *(dest+i) = CheckCell(m_matrix, i);
        }
}

#endif 
/* end non-SSE block */
