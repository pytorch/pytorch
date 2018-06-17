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

#include "pover_global.h"  // for declaration of DEFINE and INIT

DEFINE Polygon_map_t *gPolymap1 INIT(0);
DEFINE Polygon_map_t *gPolymap2 INIT(0);
DEFINE Polygon_map_t *gResultMap INIT(0);

extern void Usage(int argc, char **argv);

extern bool ParseCmdLine(int argc, char **argv );

extern bool GenerateMap(Polygon_map_t **newMap, int xSize, int ySize, int gNPolygons, colorcomp_t maxR, colorcomp_t maxG, colorcomp_t maxB);

extern bool PolygonsOverlap(RPolygon *p1, RPolygon *p2, int &xl, int &yl, int &xh, int &yh);

extern void CheckPolygonMap(Polygon_map_t *checkMap);

extern bool CompOnePolygon(RPolygon *p1, RPolygon *p2);

extern bool PolygonsEqual(RPolygon *p1, RPolygon *p2);

extern bool ComparePolygonMaps(Polygon_map_t *map1, Polygon_map_t *map2);

extern void SetRandomSeed(int newSeed);

extern int NextRan(int n);
