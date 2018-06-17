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

// Polygon overlay
//
#include <iostream>
#include <algorithm>
#include <string.h>
#include <cstdlib>
#include <assert.h>
#include "tbb/tick_count.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/mutex.h"
#include "tbb/spin_mutex.h"
#include "polyover.h"
#include "polymain.h"
#include "pover_video.h"

using namespace std;

/*!
* @brief intersects a polygon with a map, adding any results to output map
*
* @param[out] resultMap output map (must be allocated)
* @param[in] polygon to be intersected
* @param[in] map intersected against
* @param[in] lock to use when adding output polygons to result map
*
*/
void OverlayOnePolygonWithMap(Polygon_map_t *resultMap, RPolygon *myPoly, Polygon_map_t *map2, tbb::spin_mutex *rMutex) {
    int r1, g1, b1, r2, g2, b2;
    int myr=0;
    int myg=0;
    int myb=0;
    int p1Area = myPoly->area();
    for(unsigned int j=1; (j < map2->size()) && (p1Area > 0); j++) {
        RPolygon *p2 = &((*map2)[j]);
        RPolygon *pnew;
        int newxMin, newxMax, newyMin, newyMax;
        myPoly->getColor(&r1, &g1, &b1);
        if(PolygonsOverlap(myPoly, p2, newxMin, newyMin, newxMax, newyMax)) {
            p2->getColor(&r2, &g2, &b2);
            myr = r1 + r2;
            myg = g1 + g2;
            myb = b1 + b2;
            p1Area -= (newxMax-newxMin+1)*(newyMax - newyMin + 1);
            if(rMutex) {
                tbb::spin_mutex::scoped_lock lock(*rMutex);
                resultMap->push_back(RPolygon(newxMin, newyMin, newxMax, newyMax, myr, myg, myb));
            }
            else {
                resultMap->push_back(RPolygon(newxMin, newyMin, newxMax, newyMax, myr, myg, myb));
            }
        }
    }
}

/*!
* @brief Serial version of polygon overlay
* @param[out] output map
* @param[in]  first map (map that individual polygons are taken from)
* @param[in]  second map (map passed to OverlayOnePolygonWithMap)
*/
void SerialOverlayMaps(Polygon_map_t **resultMap, Polygon_map_t *map1, Polygon_map_t *map2) {
    cout << "SerialOverlayMaps called" << std::endl;
    *resultMap = new Polygon_map_t;

    RPolygon *p0 = &((*map1)[0]);
    int mapxSize, mapySize, ignore1, ignore2;
    p0->get(&ignore1, &ignore2, &mapxSize, &mapySize);
    (*resultMap)->reserve(mapxSize*mapySize); // can't be any bigger than this
    // push the map size as the first polygon,
    (*resultMap)->push_back(RPolygon(0,0,mapxSize, mapySize));
    for(unsigned int i=1; i < map1->size(); i++) {
        RPolygon *p1 = &((*map1)[i]);
        OverlayOnePolygonWithMap(*resultMap, p1, map2, NULL);
    }
}

/*!
* @class ApplyOverlay
* @brief Simple version of parallel overlay (make parallel on polygons in map1)
*/
class ApplyOverlay {
    Polygon_map_t *m_map1, *m_map2, *m_resultMap;
    tbb::spin_mutex *m_rMutex;
public:
    /*!
    * @brief functor to apply
    * @param[in] r range of polygons to intersect from map1
    */
    void operator()( const tbb::blocked_range<int> & r) const {
        PRINT_DEBUG("From " << r.begin() << " to " << r.end());
        for(int i=r.begin(); i != r.end(); i++) {
            RPolygon *myPoly = &((*m_map1)[i]);
            OverlayOnePolygonWithMap(m_resultMap, myPoly, m_map2, m_rMutex);
        }
    }
    ApplyOverlay(Polygon_map_t *resultMap, Polygon_map_t *map1, Polygon_map_t *map2, tbb::spin_mutex *rmutex) :
    m_resultMap(resultMap), m_map1(map1), m_map2(map2), m_rMutex(rmutex) {}
};

/*!
* @brief apply the parallel algorithm
* @param[out] result_map generated map
* @param[in] polymap1 first map to be applied (algorithm is parallel on this map)
* @param[in] polymap2 second map.
*/
void NaiveParallelOverlay(Polygon_map_t *&result_map, Polygon_map_t &polymap1, Polygon_map_t &polymap2) {
// -----------------------------------
    bool automatic_threadcount = false;

    if(gThreadsLow == THREADS_UNSET || gThreadsLow == tbb::task_scheduler_init::automatic) {
        gThreadsLow = gThreadsHigh = tbb::task_scheduler_init::automatic;
        automatic_threadcount = true;
    }
    result_map = new Polygon_map_t;

    RPolygon *p0 = &(polymap1[0]);
    int mapxSize, mapySize, ignore1, ignore2;
    p0->get(&ignore1, &ignore2, &mapxSize, &mapySize);
    result_map->reserve(mapxSize*mapySize); // can't be any bigger than this
    // push the map size as the first polygon,
    tbb::spin_mutex *resultMutex = new tbb::spin_mutex();
    int grain_size = gGrainSize;

    for(int nthreads = gThreadsLow; nthreads <= gThreadsHigh; nthreads++) {
        tbb::task_scheduler_init init(nthreads);
        if(gIsGraphicalVersion) {
            RPolygon *xp = new RPolygon(0, 0, gMapXSize-1, gMapYSize-1, 0, 0, 0);  // Clear the output space
            delete xp;
        }
        // put size polygon in result map
        result_map->push_back(RPolygon(0,0,mapxSize, mapySize));

        tbb::tick_count t0 = tbb::tick_count::now();
        tbb::parallel_for (tbb::blocked_range<int>(1,(int)(polymap1.size()),grain_size), ApplyOverlay(result_map, &polymap1, &polymap2, resultMutex));
        tbb::tick_count t1 = tbb::tick_count::now();

        double naiveParallelTime = (t1-t0).seconds() * 1000;
        cout << "Naive parallel with spin lock and ";
        if(automatic_threadcount) cout << "automatic";
        else cout << nthreads;
        cout << ((nthreads == 1) ? " thread" : " threads");
        cout << " took " << naiveParallelTime << " msec : speedup over serial " << (gSerialTime / naiveParallelTime) << std::endl;
        if(gCsvFile.is_open()) {
            gCsvFile << "," << naiveParallelTime;
        }
#if _DEBUG
        CheckPolygonMap(result_map);
        ComparePolygonMaps(result_map, gResultMap);
#endif
        result_map->clear();
    }
    delete resultMutex;
    if(gCsvFile.is_open()) {
        gCsvFile << std::endl;
    }
// -----------------------------------
}

template<typename T>
void split_at( Flagged_map_t& in_map, Flagged_map_t &left_out, Flagged_map_t &right_out, const T median) {
    left_out.reserve(in_map.size());
    right_out.reserve(in_map.size());
    for(Flagged_map_t::iterator i = in_map.begin(); i != in_map.end(); ++i ) {
        RPolygon *p = i->p();
        if(p->xmax() < median) {
            // in left map
            left_out.push_back(*i);
        }
        else if(p->xmin() >= median) {
            right_out.push_back(*i);
            // in right map
        }
        else {
            // in both maps.
            left_out.push_back(*i);
            right_out.push_back(RPolygon_flagged(p, true));
        }
    }
}

// range that splits the maps as well as the range.  the flagged_map_t are
// vectors of pointers, and each range owns its maps (has to free them on destruction.)
template <typename T>
class blocked_range_with_maps {
    
    typedef blocked_range<T> my_range_type;

private:

    my_range_type my_range;
    Flagged_map_t my_map1;
    Flagged_map_t my_map2;

public:

    blocked_range_with_maps(
            T begin, T end, typename my_range_type::size_type my_grainsize,
            Polygon_map_t *p1, Polygon_map_t *p2
            )
        : my_range(begin, end, my_grainsize)
    {
        my_map1.reserve(p1->size());
        my_map2.reserve(p2->size());
        for(int i=1; i < p1->size(); ++i) {
            my_map1.push_back(RPolygon_flagged(&((*p1)[i]), false));
        }
        for(int i=1; i < p2->size(); ++i) {
            my_map2.push_back(RPolygon_flagged(&(p2->at(i)), false));
        }
    }

    // copy-constructor required for deep copy of flagged maps.  One copy is done at the start of the
    // parallel for.
    blocked_range_with_maps(const blocked_range_with_maps& other): my_range(other.my_range), my_map1(other.my_map1), my_map2(other.my_map2) { }
    bool empty() const { return my_range.empty(); }
    bool is_divisible() const { return my_range.is_divisible(); }

#if _DEBUG
    void check_my_map() {
        assert(my_range.begin() <= my_range.end());
        for(Flagged_map_t::iterator ci = my_map1.begin(); ci != my_map1.end(); ++ci) {
            RPolygon *rp = ci->p();
            assert(rp->xmax() >= my_range.begin());
            assert(rp->xmin() < my_range.end());
        }
        for(Flagged_map_t::iterator ci = my_map2.begin(); ci != my_map2.end(); ++ci) {
            RPolygon *rp = ci->p();
            assert(rp->xmax() >= my_range.begin());
            assert(rp->xmin() < my_range.end());
        }
    }

    void dump_map( Flagged_map_t& mapx) {
        cout << " ** MAP **\n";
        for( Flagged_map_t::iterator ci = mapx.begin(); ci != mapx.end(); ++ci) {
            cout << *(ci->p());
            if(ci->isDuplicate()) {
                cout << " -- is_duplicate";
            }
            cout << "\n";
        }
        cout << "\n";
    }
#endif

    blocked_range_with_maps(blocked_range_with_maps& lhs_r, split ) : my_range(my_range_type(lhs_r.my_range, split())) {
        // lhs_r.my_range makes my_range from [median, high) and rhs_r.my_range from [low, median)
        Flagged_map_t original_map1 = lhs_r.my_map1;
        Flagged_map_t original_map2 = lhs_r.my_map2;
        lhs_r.my_map1.clear();
        lhs_r.my_map2.clear();
        split_at(original_map1, lhs_r.my_map1, my_map1, my_range.begin());
        split_at(original_map2, lhs_r.my_map2, my_map2, my_range.begin());
#if _DEBUG
        this->check_my_map();
        lhs_r.check_my_map();
#endif
    }

    const my_range_type& range() const { return my_range; }
    Flagged_map_t& map1() { return my_map1; }
    Flagged_map_t& map2() { return my_map2; }
};

/*!
* @class ApplySplitOverlay
* @brief parallel by columnar strip
*/
class ApplySplitOverlay {
    Polygon_map_t *m_map1, *m_map2, *m_resultMap;
    tbb::spin_mutex *m_rMutex;
public:
    /*!
    * @brief functor for columnar parallel version
    * @param[in] r range of map to be operated on
    */
    void operator()(/*const*/ blocked_range_with_maps<int> & r) const {
#ifdef _DEBUG
        // if we are debugging, serialize the method.  That way we can
        // see what is happening in each strip without the interleaving
        // confusing things.
        tbb::spin_mutex::scoped_lock lock(*m_rMutex);
        cout << unitbuf << "From " << r.range().begin() << " to " << r.range().end()-1 << std::endl;
#endif
        // get yMapSize
        int r1, g1, b1, r2, g2, b2;
        int myr=-1;
        int myg=-1;
        int myb=-1;
        int i1, i2, i3, yMapSize;
        (*m_map1)[0].get(&i1, &i2, &i3, &yMapSize);

        Flagged_map_t &fmap1 = r.map1();
        Flagged_map_t &fmap2 = r.map2();

        // When intersecting polygons from fmap1 and fmap2, if BOTH are flagged
        // as duplicate, don't add the result to the output map.  We can still
        // intersect them, because we are keeping track of how much of the polygon
        // is left over from intersecting, and quitting when the polygon is
        // used up.

        for(unsigned int ii=0; ii < fmap1.size(); ii++) {
            RPolygon *p1 = fmap1[ii].p();
            bool is_dup = fmap1[ii].isDuplicate();
            int parea = p1->area();
            p1->getColor(&r1, &g1, &b1);
            for(unsigned int jj=0;(jj < fmap2.size()) && (parea > 0); jj++) {
                int xl, yl, xh, yh;
                RPolygon *p2 = fmap2[jj].p();
                if(PolygonsOverlap(p1, p2, xl, yl, xh, yh)) {
                    if(!(is_dup && fmap2[jj].isDuplicate())) {
                        p2->getColor(&r2, &g2, &b2);
                        myr = r1 + r2;
                        myg = g1 + g2;
                        myb = b1 + b2;
#ifdef _DEBUG
#else
                        tbb::spin_mutex::scoped_lock lock(*m_rMutex);
#endif
                        (*m_resultMap).push_back(RPolygon(xl, yl, xh, yh, myr, myg, myb));
                    }
                    parea -= (xh-xl+1)*(yh-yl+1);
                }
            }
        }
    }

    ApplySplitOverlay(Polygon_map_t *resultMap, Polygon_map_t *map1, Polygon_map_t *map2, tbb::spin_mutex *rmutex) :
    m_resultMap(resultMap), m_map1(map1), m_map2(map2), m_rMutex(rmutex) {}
};


/*!
* @brief intersects two maps strip-wise
*
* @param[out] resultMap output map (must be allocated)
* @param[in] polymap1 map to be intersected
* @param[in] polymap2 map to be intersected
*/
void SplitParallelOverlay(Polygon_map_t **result_map, Polygon_map_t *polymap1, Polygon_map_t *polymap2) {
    int nthreads;
    bool automatic_threadcount = false;
    double domainSplitParallelTime;
    tbb::tick_count t0, t1;
    tbb::spin_mutex *resultMutex;
    if(gThreadsLow == THREADS_UNSET || gThreadsLow == tbb::task_scheduler_init::automatic ) {
        gThreadsLow = gThreadsHigh = tbb::task_scheduler_init::automatic;
        automatic_threadcount = true;
    }
    *result_map = new Polygon_map_t;

    RPolygon *p0 = &((*polymap1)[0]);
    int mapxSize, mapySize, ignore1, ignore2;
    p0->get(&ignore1, &ignore2, &mapxSize, &mapySize);
    (*result_map)->reserve(mapxSize*mapySize); // can't be any bigger than this
    resultMutex = new tbb::spin_mutex();

    int grain_size;
#ifdef _DEBUG
    grain_size = gMapXSize / 4;
#else
    grain_size = gGrainSize;
#endif
    for(nthreads = gThreadsLow; nthreads <= gThreadsHigh; nthreads++) {
        tbb::task_scheduler_init init(nthreads);
        if(gIsGraphicalVersion) {
            RPolygon *xp = new RPolygon(0, 0, gMapXSize-1, gMapYSize-1, 0, 0, 0);  // Clear the output space
            delete xp;
        }
        // push the map size as the first polygon,
        (*result_map)->push_back(RPolygon(0,0,mapxSize, mapySize));
        t0 = tbb::tick_count::now();
        tbb::parallel_for (blocked_range_with_maps<int>(0,(int)(mapxSize+1),grain_size, polymap1, polymap2), ApplySplitOverlay((*result_map), polymap1, polymap2, resultMutex));
        t1 = tbb::tick_count::now();
        domainSplitParallelTime = (t1-t0).seconds()*1000;
        cout << "Splitting parallel with spin lock and ";
        if(automatic_threadcount) cout << "automatic";
        else cout << nthreads;
        cout << ((nthreads == 1) ? " thread" : " threads");
        cout << " took " << domainSplitParallelTime <<  " msec : speedup over serial " << (gSerialTime / domainSplitParallelTime) << std::endl;
        if(gCsvFile.is_open()) {
            gCsvFile << "," << domainSplitParallelTime;
        }
#if _DEBUG
        CheckPolygonMap(*result_map);
        ComparePolygonMaps(*result_map, gResultMap);
#endif
        (*result_map)->clear();

    }
    delete resultMutex;
    if(gCsvFile.is_open()) {
        gCsvFile << std::endl;
    }
}

class ApplySplitOverlayCV {
    Polygon_map_t *m_map1, *m_map2;
    concurrent_Polygon_map_t *m_resultMap;
public:
    /*!
    * @brief functor for columnar parallel version
    * @param[in] r range of map to be operated on
    */
    void operator()(blocked_range_with_maps<int> & r) const {
        // get yMapSize
        int r1, g1, b1, r2, g2, b2;
        int myr=-1;
        int myg=-1;
        int myb=-1;
        int i1, i2, i3, yMapSize;
        (*m_map1)[0].get(&i1, &i2, &i3, &yMapSize);

        Flagged_map_t &fmap1 = r.map1();
        Flagged_map_t &fmap2 = r.map2();

        // When intersecting polygons from fmap1 and fmap2, if BOTH are flagged
        // as duplicate, don't add the result to the output map.  We can still
        // intersect them, because we are keeping track of how much of the polygon
        // is left over from intersecting, and quitting when the polygon is
        // used up.

        for(unsigned int ii=0; ii < fmap1.size(); ii++) {
            RPolygon *p1 = fmap1[ii].p();
            bool is_dup = fmap1[ii].isDuplicate();
            int parea = p1->area();
            p1->getColor(&r1, &g1, &b1);
            for(unsigned int jj=0;(jj < fmap2.size()) && (parea > 0); jj++) {
                int xl, yl, xh, yh;
                RPolygon *p2 = fmap2[jj].p();
                if(PolygonsOverlap(p1, p2, xl, yl, xh, yh)) {
                    if(!(is_dup && fmap2[jj].isDuplicate())) {
                        p2->getColor(&r2, &g2, &b2);
                        myr = r1 + r2;
                        myg = g1 + g2;
                        myb = b1 + b2;
                        (*m_resultMap).push_back(RPolygon(xl, yl, xh, yh, myr, myg, myb));
                    }
                    parea -= (xh-xl+1)*(yh-yl+1);
                }
            }
        }
    }

    ApplySplitOverlayCV(concurrent_Polygon_map_t *resultMap, Polygon_map_t *map1, Polygon_map_t *map2 ) :
    m_resultMap(resultMap), m_map1(map1), m_map2(map2) {}
};


/*!
* @brief intersects two maps strip-wise, accumulating into a concurrent_vector
*
* @param[out] resultMap output map (must be allocated)
* @param[in] polymap1 map to be intersected
* @param[in] polymap2 map to be intersected
*/
void SplitParallelOverlayCV(concurrent_Polygon_map_t **result_map, Polygon_map_t *polymap1, Polygon_map_t *polymap2) {
    int nthreads;
    bool automatic_threadcount = false;
    double domainSplitParallelTime;
    tbb::tick_count t0, t1;
    if(gThreadsLow == THREADS_UNSET || gThreadsLow == tbb::task_scheduler_init::automatic ) {
        gThreadsLow = gThreadsHigh = tbb::task_scheduler_init::automatic;
        automatic_threadcount = true;
    }
    *result_map = new concurrent_Polygon_map_t;

    RPolygon *p0 = &((*polymap1)[0]);
    int mapxSize, mapySize, ignore1, ignore2;
    p0->get(&ignore1, &ignore2, &mapxSize, &mapySize);
    // (*result_map)->reserve(mapxSize*mapySize); // can't be any bigger than this

    int grain_size;
#ifdef _DEBUG
    grain_size = gMapXSize / 4;
#else
    grain_size = gGrainSize;
#endif
    for(nthreads = gThreadsLow; nthreads <= gThreadsHigh; nthreads++) {
        tbb::task_scheduler_init init(nthreads);
        if(gIsGraphicalVersion) {
            RPolygon *xp = new RPolygon(0, 0, gMapXSize-1, gMapYSize-1, 0, 0, 0);  // Clear the output space
            delete xp;
        }
        // push the map size as the first polygon,
        (*result_map)->push_back(RPolygon(0,0,mapxSize, mapySize));
        t0 = tbb::tick_count::now();
        tbb::parallel_for (blocked_range_with_maps<int>(0,(int)(mapxSize+1),grain_size, polymap1, polymap2), ApplySplitOverlayCV((*result_map), polymap1, polymap2));
        t1 = tbb::tick_count::now();
        domainSplitParallelTime = (t1-t0).seconds()*1000;
        cout << "Splitting parallel with concurrent_vector and ";
        if(automatic_threadcount) cout << "automatic";
        else cout << nthreads;
        cout << ((nthreads == 1) ? " thread" : " threads");
        cout << " took " << domainSplitParallelTime <<  " msec : speedup over serial " << (gSerialTime / domainSplitParallelTime) << std::endl;
        if(gCsvFile.is_open()) {
            gCsvFile << "," << domainSplitParallelTime;
        }
#if _DEBUG
        {
            
            Polygon_map_t s_result_map;
            for(concurrent_Polygon_map_t::const_iterator ci = (*result_map)->begin(); ci != (*result_map)->end(); ++ci) {
                s_result_map.push_back(*ci);
            }
            CheckPolygonMap(&s_result_map);
            ComparePolygonMaps(&s_result_map, gResultMap);
        }
#endif
        (*result_map)->clear();

    }

    if(gCsvFile.is_open()) {
        gCsvFile << std::endl;
    }

}

// ------------------------------------------------------

class ApplySplitOverlayETS {
    Polygon_map_t *m_map1, *m_map2;
    ETS_Polygon_map_t *m_resultMap;
public:
    /*!
    * @brief functor for columnar parallel version
    * @param[in] r range of map to be operated on
    */
    void operator()(blocked_range_with_maps<int> & r) const {
        // get yMapSize
        int r1, g1, b1, r2, g2, b2;
        int myr=-1;
        int myg=-1;
        int myb=-1;
        int i1, i2, i3, yMapSize;
        (*m_map1)[0].get(&i1, &i2, &i3, &yMapSize);

        Flagged_map_t &fmap1 = r.map1();
        Flagged_map_t &fmap2 = r.map2();

        // When intersecting polygons from fmap1 and fmap2, if BOTH are flagged
        // as duplicate, don't add the result to the output map.  We can still
        // intersect them, because we are keeping track of how much of the polygon
        // is left over from intersecting, and quitting when the polygon is
        // used up.

        for(unsigned int ii=0; ii < fmap1.size(); ii++) {
            RPolygon *p1 = fmap1[ii].p();
            bool is_dup = fmap1[ii].isDuplicate();
            int parea = p1->area();
            p1->getColor(&r1, &g1, &b1);
            for(unsigned int jj=0;(jj < fmap2.size()) && (parea > 0); jj++) {
                int xl, yl, xh, yh;
                RPolygon *p2 = fmap2[jj].p();
                if(PolygonsOverlap(p1, p2, xl, yl, xh, yh)) {
                    if(!(is_dup && fmap2[jj].isDuplicate())) {
                        p2->getColor(&r2, &g2, &b2);
                        myr = r1 + r2;
                        myg = g1 + g2;
                        myb = b1 + b2;
                        (*m_resultMap).local().push_back(RPolygon(xl, yl, xh, yh, myr, myg, myb));
                    }
                    parea -= (xh-xl+1)*(yh-yl+1);
                }
            }
        }
    }

    ApplySplitOverlayETS(ETS_Polygon_map_t *resultMap, Polygon_map_t *map1, Polygon_map_t *map2 ) :
    m_resultMap(resultMap), m_map1(map1), m_map2(map2) {}
};


/*!
* @brief intersects two maps strip-wise, accumulating into an ets variable
*
* @param[out] resultMap output map (must be allocated)
* @param[in] polymap1 map to be intersected
* @param[in] polymap2 map to be intersected
*/
void SplitParallelOverlayETS(ETS_Polygon_map_t **result_map, Polygon_map_t *polymap1, Polygon_map_t *polymap2) {
    int nthreads;
    bool automatic_threadcount = false;
    double domainSplitParallelTime;
    tbb::tick_count t0, t1;
    if(gThreadsLow == THREADS_UNSET || gThreadsLow == tbb::task_scheduler_init::automatic ) {
        gThreadsLow = gThreadsHigh = tbb::task_scheduler_init::automatic;
        automatic_threadcount = true;
    }
    *result_map = new ETS_Polygon_map_t;

    RPolygon *p0 = &((*polymap1)[0]);
    int mapxSize, mapySize, ignore1, ignore2;
    p0->get(&ignore1, &ignore2, &mapxSize, &mapySize);
    // (*result_map)->reserve(mapxSize*mapySize); // can't be any bigger than this

    int grain_size;
#ifdef _DEBUG
    grain_size = gMapXSize / 4;
#else
    grain_size = gGrainSize;
#endif
    for(nthreads = gThreadsLow; nthreads <= gThreadsHigh; nthreads++) {
        tbb::task_scheduler_init init(nthreads);
        if(gIsGraphicalVersion) {
            RPolygon *xp = new RPolygon(0, 0, gMapXSize-1, gMapYSize-1, 0, 0, 0);  // Clear the output space
            delete xp;
        }
        // push the map size as the first polygon,
        // This polygon needs to be first, so we can push it at the start of a combine.
        // (*result_map)->local.push_back(RPolygon(0,0,mapxSize, mapySize));
        t0 = tbb::tick_count::now();
        tbb::parallel_for (blocked_range_with_maps<int>(0,(int)(mapxSize+1),grain_size, polymap1, polymap2), ApplySplitOverlayETS((*result_map), polymap1, polymap2));
        t1 = tbb::tick_count::now();
        domainSplitParallelTime = (t1-t0).seconds()*1000;
        cout << "Splitting parallel with ETS and ";
        if(automatic_threadcount) cout << "automatic";
        else cout << nthreads;
        cout << ((nthreads == 1) ? " thread" : " threads");
        cout << " took " << domainSplitParallelTime <<  " msec : speedup over serial " << (gSerialTime / domainSplitParallelTime) << std::endl;
        if(gCsvFile.is_open()) {
            gCsvFile << "," << domainSplitParallelTime;
        }
#if _DEBUG
        {
            
            Polygon_map_t s_result_map;
            flattened2d<ETS_Polygon_map_t> psv = flatten2d(**result_map);
            s_result_map.push_back(RPolygon(0,0,mapxSize, mapySize));
            for(flattened2d<ETS_Polygon_map_t>::const_iterator ci = psv.begin(); ci != psv.end(); ++ci) {
                s_result_map.push_back(*ci);
            }
            CheckPolygonMap(&s_result_map);
            ComparePolygonMaps(&s_result_map, gResultMap);
        }
#endif
        (*result_map)->clear();

    }

    if(gCsvFile.is_open()) {
        gCsvFile << std::endl;
    }

}
