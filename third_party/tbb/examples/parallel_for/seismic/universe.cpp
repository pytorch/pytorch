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

#include "../../common/gui/video.h"
#include <cmath>
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"


using namespace std;

#ifdef _MSC_VER
// warning C4068: unknown pragma
#pragma warning(disable: 4068)
// warning C4351: new behavior: elements of array 'array' will be default initialized
#pragma warning(disable: 4351)
#endif

#include "universe.h"

const colorcomp_t MaterialColor[4][3] = { // BGR
    {96,0,0},     // WATER
    {0,48,48},    // SANDSTONE
    {32,32,23}    // SHALE
};

void Universe::InitializeUniverse(video const& colorizer) {

    pulseCounter = pulseTime = 100;
    pulseX = UniverseWidth/3;
    pulseY = UniverseHeight/4;
    // Initialize V, S, and T to slightly non-zero values, in order to avoid denormal waves.
    for( int i=0; i<UniverseHeight; ++i )
#pragma ivdep
        for( int j=0; j<UniverseWidth; ++j ) {
            T[i][j] = S[i][j] = V[i][j] = ValueType(1.0E-6);
        }
    for( int i=1; i<UniverseHeight-1; ++i ) {
        for( int j=1; j<UniverseWidth-1; ++j ) {
            float x = float(j-UniverseWidth/2)/(UniverseWidth/2);
            ValueType t = (ValueType)i/UniverseHeight;
            MaterialType m;
            D[i][j] = 1.0;
            // Coefficient values are fictitious, and chosen to visually exaggerate
            // physical effects such as Rayleigh waves.  The fabs/exp line generates
            // a shale layer with a gentle upwards slope and an anticline.
            if( t<0.3f ) {
                m = WATER;
                M[i][j] = 0.125;
                L[i][j] = 0.125;
            } else if( fabs(t-0.7+0.2*exp(-8*x*x)+0.025*x)<=0.1 ) {
                m = SHALE;
                M[i][j] = 0.5;
                L[i][j] = 0.6;
            } else {
                m = SANDSTONE;
                M[i][j] = 0.3;
                L[i][j] = 0.4;
            }
            material[i][j] = m;
        }
    }
    ValueType scale = 2.0f/ColorMapSize;
    for( int k=0; k<4; ++k ) {
        for( int i=0; i<ColorMapSize; ++i ) {
            colorcomp_t c[3];
            ValueType t = (i-ColorMapSize/2)*scale;
            ValueType r = t>0 ? t : 0;
            ValueType b = t<0 ? -t : 0;
            ValueType g = 0.5f*fabs(t);
            memcpy(c, MaterialColor[k], sizeof(c));
            c[2] = colorcomp_t(r*(255-c[2])+c[2]);
            c[1] = colorcomp_t(g*(255-c[1])+c[1]);
            c[0] = colorcomp_t(b*(255-c[0])+c[0]);
            ColorMap[k][i] = colorizer.get_color(c[2], c[1], c[0]);
        }
    }
    // Set damping coefficients around border to reduce reflections from boundaries.
    ValueType d = 1.0;
    for( int k=DamperSize-1; k>0; --k ) {
        d *= 1-1.0f/(DamperSize*DamperSize);
        for( int j=1; j<UniverseWidth-1; ++j ) {
            D[k][j] *= d;
            D[UniverseHeight-1-k][j] *= d;
        }
        for( int i=1; i<UniverseHeight-1; ++i ) {
            D[i][k] *= d;
            D[i][UniverseWidth-1-k] *= d;
        }
    }
    drawingMemory = colorizer.get_drawing_memory();
}
void Universe::UpdatePulse() {
    if( pulseCounter>0 ) {
        ValueType t = (pulseCounter-pulseTime/2)*0.05f;
        V[pulseY][pulseX] += 64*sqrt(M[pulseY][pulseX])*exp(-t*t);
        --pulseCounter;
    }
}

struct Universe::Rectangle {
    struct std::pair<int,int> xRange;
    struct std::pair<int,int> yRange;
    Rectangle (int startX, int startY, int width, int height):xRange(startX,width),yRange(startY,height){}
    int StartX() const {return xRange.first;}
    int StartY() const {return yRange.first;}
    int Width()   const {return xRange.second;}
    int Height()  const {return yRange.second;}
    int EndX() const {return xRange.first + xRange.second;}
    int EndY() const {return yRange.first + yRange.second;}

};

void Universe::UpdateStress(Rectangle const& r ) {
    drawing_area  drawing(r.StartX(),r.StartY(),r.Width(),r.Height(),drawingMemory);
    for( int i=r.StartY(); i<r.EndY() ; ++i ) {
        drawing.set_pos(1, i-r.StartY());
#pragma ivdep
        for( int j=r.StartX(); j<r.EndX() ; ++j ) {
            S[i][j] += M[i][j]*(V[i][j+1]-V[i][j]);
            T[i][j] += M[i][j]*(V[i+1][j]-V[i][j]);
            int index = (int)(V[i][j]*(ColorMapSize/2)) + ColorMapSize/2;
            if( index<0 ) index = 0;
            if( index>=ColorMapSize ) index = ColorMapSize-1;
            color_t* c = ColorMap[material[i][j]];
            drawing.put_pixel(c[index]);
        }
    }
}

void Universe::SerialUpdateStress() {
    Rectangle  area(0, 0, UniverseWidth-1, UniverseHeight-1);
    UpdateStress(area);
}

struct UpdateStressBody {
    Universe & u_;
    UpdateStressBody(Universe & u):u_(u){}
    void operator()( const tbb::blocked_range<int>& range ) const {
        Universe::Rectangle area(0, range.begin(), u_.UniverseWidth-1, range.size());
        u_.UpdateStress(area);
    }
};

void Universe::ParallelUpdateStress(tbb::affinity_partitioner &affinity) {
    tbb::parallel_for( tbb::blocked_range<int>( 0, UniverseHeight-1 ), // Index space for loop
                       UpdateStressBody(*this),                             // Body of loop
                       affinity );                                     // Affinity hint
}

void Universe::UpdateVelocity(Rectangle const& r) {
    for( int i=r.StartY(); i<r.EndY(); ++i )
#pragma ivdep
        for( int j=r.StartX(); j<r.EndX(); ++j )
            V[i][j] = D[i][j]*(V[i][j] + L[i][j]*(S[i][j] - S[i][j-1] + T[i][j] - T[i-1][j]));
}

void Universe::SerialUpdateVelocity() {
    UpdateVelocity(Rectangle(1,1,UniverseWidth-1,UniverseHeight-1));
}

struct UpdateVelocityBody {
    Universe & u_;
    UpdateVelocityBody(Universe & u):u_(u){}
    void operator()( const tbb::blocked_range<int>& y_range ) const {
        u_.UpdateVelocity(Universe::Rectangle(1,y_range.begin(),u_.UniverseWidth-1,y_range.size()));
    }
};

void Universe::ParallelUpdateVelocity(tbb::affinity_partitioner &affinity) {
    tbb::parallel_for( tbb::blocked_range<int>( 1, UniverseHeight ), // Index space for loop
                       UpdateVelocityBody(*this),                    // Body of loop
                       affinity );                                   // Affinity hint
}

void Universe::SerialUpdateUniverse() {
    UpdatePulse();
    SerialUpdateStress();
    SerialUpdateVelocity();
}

void Universe::ParallelUpdateUniverse() {
    /** Affinity is an argument to parallel_for to hint that an iteration of a loop
    is best replayed on the same processor for each execution of the loop.
    It is a static object because it must remember where the iterations happened
    in previous executions. */
    static tbb::affinity_partitioner affinity;
    UpdatePulse();
    ParallelUpdateStress(affinity);
    ParallelUpdateVelocity(affinity);
}

bool Universe::TryPutNewPulseSource(int x, int y){
    if(pulseCounter == 0) {
        pulseCounter = pulseTime;
        pulseX = x; pulseY = y;
        return true;
    }
    return false;
}

void Universe::SetDrawingMemory(const drawing_memory &dmem) {
    drawingMemory = dmem;
}
