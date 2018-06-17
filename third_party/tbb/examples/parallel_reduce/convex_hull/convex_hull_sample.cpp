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

/*
    This file contains the TBB-based implementation of convex hull algortihm.
    It corresponds to the following settings in convex_hull_bench.cpp:
    - USETBB defined to 1
    - USECONCVEC defined to 1
    - INIT_ONCE defined to 0
    - only buffered version is used
*/
#include "convex_hull.h"

#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"
#include "tbb/tick_count.h"
#include "tbb/concurrent_vector.h"

typedef util::point<double>               point_t;
typedef tbb::concurrent_vector< point_t > pointVec_t;
typedef tbb::blocked_range<size_t>        range_t;

void appendVector(const point_t* src, size_t srcSize, pointVec_t& dest) {
    std::copy(src, src + srcSize, dest.grow_by(srcSize));
}

void appendVector(const pointVec_t& src, pointVec_t& dest) {
    std::copy(src.begin(), src.end(), dest.grow_by(src.size()));
}
class FillRNDPointsVector_buf {
    pointVec_t          &points;
public:
    static const size_t  grainSize = cfg::generateGrainSize;

    explicit FillRNDPointsVector_buf(pointVec_t& _points)
        : points(_points) {}

    void operator()(const range_t& range) const {
        util::rng the_rng(range.begin());
        const size_t i_end = range.end();
        size_t count = 0, j = 0;
        point_t tmp_vec[grainSize];

        for(size_t i=range.begin(); i!=i_end; ++i) {
            tmp_vec[j++] = util::GenerateRNDPoint<double>(count, the_rng, util::rng::max_rand);
        }
        //Here we have race condition. Elements being written to may be still under construction.
        //For C++ 2003 it is workarounded by vector element type which default constructor does not touch memory,
        //it being constructed on. See comments near default ctor of point class for more details.
        //Strictly speaking it is UB.
        //TODO: need to find more reliable/correct way
        points.grow_to_at_least(range.end());
        std::copy(tmp_vec, tmp_vec+j,points.begin()+range.begin());
    }
};

void initialize(pointVec_t &points) {
    //This function generate the same series of point on every call.
    //Reproducibility is needed for benchmarking to produce reliable results.
    //It is achieved through the following points:
    //      - FillRNDPointsVector_buf instance has its own local instance
    //        of random number generator, which in turn does not use any global data
    //      - tbb::simple_partitioner produce the same set of ranges on every call to
    //        tbb::parallel_for
    //      - local RNG instances are seeded by the starting indexes of corresponding ranges
    //      - grow_to_at_least() enables putting points into the resulting vector in deterministic order
    //        (unlike concurrent push_back or grow_by).

    // In the buffered version, a temporary storage for as much as grainSize elements 
    // is allocated inside the body. Since auto_partitioner may increase effective
    // range size which would cause a crash, simple partitioner has to be used.
    tbb::parallel_for(range_t(0, cfg::numberOfPoints, FillRNDPointsVector_buf::grainSize),
                      FillRNDPointsVector_buf(points), tbb::simple_partitioner());
}

class FindXExtremum {
public:
    typedef enum {
        minX, maxX
    } extremumType;

    static const size_t  grainSize = cfg::findExtremumGrainSize;

    FindXExtremum(const pointVec_t& points_, extremumType exType_)
        : points(points_), exType(exType_), extrXPoint(points[0]) {}

    FindXExtremum(const FindXExtremum& fxex, tbb::split)
        // Can run in parallel with fxex.operator()() or fxex.join().
        // The data race reported by tools is harmless.
        : points(fxex.points), exType(fxex.exType), extrXPoint(fxex.extrXPoint) {}

    void operator()(const range_t& range) {
        const size_t i_end = range.end();
        if(!range.empty()) {
            for(size_t i = range.begin(); i != i_end; ++i) {
                if(closerToExtremum(points[i])) {
                    extrXPoint = points[i];
                }
            }
        }
    }

    void join(const FindXExtremum &rhs) {
        if(closerToExtremum(rhs.extrXPoint)) {
            extrXPoint = rhs.extrXPoint;
        }
    }

    point_t extremeXPoint() {
        return extrXPoint;
    }

private:
    const pointVec_t    &points;
    const extremumType   exType;
    point_t              extrXPoint;
    bool closerToExtremum(const point_t &p) const {
        switch(exType) {
        case minX:
            return p.x<extrXPoint.x; break;
        case maxX:
            return p.x>extrXPoint.x; break;
        }
        return false; // avoid warning
    }
};

template <FindXExtremum::extremumType type>
point_t extremum(const pointVec_t &P) {
    FindXExtremum fxBody(P, type);
    tbb::parallel_reduce(range_t(0, P.size(), FindXExtremum::grainSize), fxBody);
    return fxBody.extremeXPoint();
}

class SplitByCP_buf {
    const pointVec_t    &initialSet;
    pointVec_t          &reducedSet;
    point_t              p1, p2;
    point_t              farPoint;
    double               howFar;
public:
    static const size_t  grainSize = cfg::divideGrainSize;

    SplitByCP_buf( point_t _p1, point_t _p2,
        const pointVec_t &_initialSet, pointVec_t &_reducedSet)
        : p1(_p1), p2(_p2),
        initialSet(_initialSet), reducedSet(_reducedSet),
        howFar(0), farPoint(p1) {}

    SplitByCP_buf(SplitByCP_buf& sbcp, tbb::split)
        : p1(sbcp.p1), p2(sbcp.p2),
        initialSet(sbcp.initialSet), reducedSet(sbcp.reducedSet),
        howFar(0), farPoint(p1) {}

    void operator()(const range_t& range) {
        const size_t i_end = range.end();
        size_t j = 0;
        double cp;
        point_t tmp_vec[grainSize];
        for(size_t i = range.begin(); i != i_end; ++i) {
            if( (initialSet[i] != p1) && (initialSet[i] != p2) ) {            
                cp = util::cross_product(p1, p2, initialSet[i]);
                if(cp>0) {
                    tmp_vec[j++] = initialSet[i];
                    if(cp>howFar) {
                        farPoint = initialSet[i];
                        howFar   = cp;
                    }
                }
            }
        }

        appendVector(tmp_vec, j, reducedSet);
    }

    void join(const SplitByCP_buf& rhs) {
        if(rhs.howFar>howFar) {
            howFar   = rhs.howFar;
            farPoint = rhs.farPoint;
        }
    }

    point_t farthestPoint() const {
        return farPoint;
    }
};

point_t divide(const pointVec_t &P, pointVec_t &P_reduced, 
                   const point_t &p1, const point_t &p2) {
    SplitByCP_buf sbcpb(p1, p2, P, P_reduced);
    // Must use simple_partitioner (see the comment in initialize() above)
    tbb::parallel_reduce(range_t(0, P.size(), SplitByCP_buf::grainSize),
                         sbcpb, tbb::simple_partitioner());

    if(util::verbose) {
        std::stringstream ss;
        ss << P.size() << " nodes in bucket"<< ", "
            << "dividing by: [ " << p1 << ", " << p2 << " ], "
            << "farthest node: " << sbcpb.farthestPoint();
        util::OUTPUT.push_back(ss.str());
    }

    return sbcpb.farthestPoint();
}

void divide_and_conquer(const pointVec_t &P, pointVec_t &H,
                            point_t p1, point_t p2) {
    assert(P.size() >= 2);
    pointVec_t P_reduced;
    pointVec_t H1, H2;
    point_t p_far = divide(P, P_reduced, p1, p2);
    if (P_reduced.size()<2) {
        H.push_back(p1);
        appendVector(P_reduced, H);
    }
    else {
        divide_and_conquer(P_reduced, H1, p1, p_far);
        divide_and_conquer(P_reduced, H2, p_far, p2);

        appendVector(H1, H);
        appendVector(H2, H);
    }
}

void quickhull(const pointVec_t &points, pointVec_t &hull) {
    if (points.size() < 2) {
        appendVector(points, hull);
        return;
    }

    point_t p_maxx = extremum<FindXExtremum::maxX>(points);
    point_t p_minx = extremum<FindXExtremum::minX>(points);

    pointVec_t H;

    divide_and_conquer(points, hull, p_maxx, p_minx);
    divide_and_conquer(points, H, p_minx, p_maxx);

    appendVector(H, hull);
}

int main(int argc, char* argv[]) {
    util::my_time_t tm_main_begin = util::gettime();

    util::ParseInputArgs(argc, argv);

    pointVec_t      points;
    pointVec_t      hull;
    int             nthreads;

    points.reserve(cfg::numberOfPoints);

    if(!util::silent) {
        std::cout << "Starting TBB-buffered version of QUICK HULL algorithm" << std::endl;
    }

    for(nthreads=cfg::threads.first; nthreads<=cfg::threads.last; nthreads=cfg::threads.step(nthreads)) {
        tbb::task_scheduler_init init(nthreads);

        points.clear();
        util::my_time_t tm_init = util::gettime();
        initialize(points);
        util::my_time_t tm_start = util::gettime();
        if(!util::silent) {
            std::cout <<"Init time on "<<nthreads<<" threads: "<<util::time_diff(tm_init, tm_start)<<"  Points in input: "<<points.size()<<std::endl;
        }

        tm_start = util::gettime();
        quickhull(points, hull);
        util::my_time_t tm_end = util::gettime();
        if(!util::silent) {
            std::cout <<"Time on "<<nthreads<<" threads: "<<util::time_diff(tm_start, tm_end)<<"  Points in hull: "<<hull.size()<<std::endl;
        }
        hull.clear();
    }
    utility::report_elapsed_time(util::time_diff(tm_main_begin, util::gettime()));
    return 0;
}
