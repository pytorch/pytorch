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

/*!
 * polyover.h : extern declarations for polyover.cpp
*/
#include "rpolygon.h"
#include "tbb/mutex.h"
#include "tbb/spin_mutex.h"

extern void OverlayOnePolygonWithMap(Polygon_map_t *resultMap, RPolygon *myPoly, Polygon_map_t  *map2, tbb::spin_mutex *rMutex);

extern void SerialOverlayMaps(Polygon_map_t **resultMap, Polygon_map_t *map1, Polygon_map_t *map2);

// extern void NaiveParallelOverlay(Polygon_map_t **result_map, Polygon_map_t *polymap1, Polygon_map_t *polymap2);
extern void NaiveParallelOverlay(Polygon_map_t *&result_map, Polygon_map_t &polymap1, Polygon_map_t &polymap2);

extern void SplitParallelOverlay(Polygon_map_t **result_map, Polygon_map_t *polymap1, Polygon_map_t *polymap2);
extern void SplitParallelOverlayCV(concurrent_Polygon_map_t **result_map, Polygon_map_t *polymap1, Polygon_map_t *polymap2);
extern void SplitParallelOverlayETS(ETS_Polygon_map_t **result_map, Polygon_map_t *polymap1, Polygon_map_t *polymap2);

extern void CheckPolygonMap(Polygon_map_t *checkMap);
extern bool ComparePolygonMaps(Polygon_map_t *map1, Polygon_map_t *map2);

