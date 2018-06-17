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

#ifndef UNIVERSE_H_
#define UNIVERSE_H_

#ifndef UNIVERSE_WIDTH
#define UNIVERSE_WIDTH 1024
#endif
#ifndef UNIVERSE_HEIGHT
#define UNIVERSE_HEIGHT 512
#endif

#include "../../common/gui/video.h"
#include "tbb/partitioner.h"

class Universe {
public:
    enum {
        UniverseWidth  = UNIVERSE_WIDTH,
        UniverseHeight = UNIVERSE_HEIGHT
    };
private:
    //in order to avoid performance degradation due to cache aliasing issue
    //some padding is needed after each row in array, and between array themselves.
    //the padding is achieved by adjusting number of rows and columns.
    //as the compiler is forced to place class members of the same clause in order of the
    //declaration this seems to be the right way of padding.

    //magic constants added below are chosen experimentally for 1024x512.
    enum {
        MaxWidth = UniverseWidth+1,
        MaxHeight = UniverseHeight+3
    };

    typedef float ValueType;

    //! Horizontal stress
    ValueType S[MaxHeight][MaxWidth];

    //! Velocity at each grid point
    ValueType V[MaxHeight][MaxWidth];

    //! Vertical stress
    ValueType T[MaxHeight][MaxWidth];

    //! Coefficient related to modulus
    ValueType M[MaxHeight][MaxWidth];

    //! Damping coefficients
    ValueType D[MaxHeight][MaxWidth];

    //! Coefficient related to lightness
    ValueType L[MaxHeight][MaxWidth];

    enum { ColorMapSize = 1024};
    color_t ColorMap[4][ColorMapSize];

    enum MaterialType {
        WATER=0,
        SANDSTONE=1,
        SHALE=2
    };

    //! Values are MaterialType, cast to an unsigned char to save space.
    unsigned char material[MaxHeight][MaxWidth];

private:
    enum { DamperSize = 32};

    int pulseTime;
    int pulseCounter;
    int pulseX;
    int pulseY;

    drawing_memory drawingMemory;

public:
    void InitializeUniverse(video const& colorizer);

    void SerialUpdateUniverse();
    void ParallelUpdateUniverse();
    bool TryPutNewPulseSource(int x, int y);
    void SetDrawingMemory(const drawing_memory &dmem);
private:
    struct Rectangle;
    void UpdatePulse();
    void UpdateStress(Rectangle const& r );

    void SerialUpdateStress() ;
    friend struct UpdateStressBody;
    friend struct UpdateVelocityBody;
    void ParallelUpdateStress(tbb::affinity_partitioner &affinity);

    void UpdateVelocity(Rectangle const& r);

    void SerialUpdateVelocity() ;
    void ParallelUpdateVelocity(tbb::affinity_partitioner &affinity);
};

#endif /* UNIVERSE_H_ */
