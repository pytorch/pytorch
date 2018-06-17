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

//
// Self-organizing map in TBB flow::graph
//
//   This is an example of the use of cancellation in a graph.  After a point in searching for
//   the best match for an example, two examples are looked for simultaneously.  When the
//   earlier example is found and the update radius is determined, the affected searches
//   for the subsequent example are cancelled, and after the update they are restarted.
//   As the update radius shrinks fewer searches are cancelled, and by the last iterations
//   virtually all the work done for the speculating example is useful.
//
// first, a simple implementation with only one example vector
// at a time.
//
// we will do a color map (the simple example.)
//
//  graph algorithm
//
//       for some number of iterations
//           update radius r, weight of change L 
//           for each example V
//               use graph to find BMU
//               for each part of map within radius of BMU W
//                   update vector:  W(t+1) = W(t) + w(dist)*L*(V - W(t))

#define _MAIN_C_ 1
#include "som.h"

#include "tbb/task_scheduler_init.h"
#include "tbb/flow_graph.h"
#include "tbb/blocked_range2d.h"
#include "tbb/tick_count.h"
#include "../../common/utility/utility.h"

#define RED 0
#define GREEN 1
#define BLUE 2

static int xranges = 1;
static int yranges = 1;
static int xsize = -1;
static int ysize = -1;

static int global_i = 0;
static int speculation_start; 
std::vector<int> function_node_execs;
static int xRangeMax = 3;
static int yRangeMax = 3;
static bool dont_speculate = false;
static search_result_type last_update;

class BMU_search_body {
    SOMap &my_map;
    subsquare_type my_square;
    int &fn_tally;
public:
    BMU_search_body(SOMap &_m, subsquare_type &_sq, int &fnt) : my_map(_m), my_square(_sq), fn_tally(fnt) { }
    BMU_search_body( const BMU_search_body &other) : my_map(other.my_map), my_square(other.my_square), fn_tally(other.fn_tally) { }
    search_result_type operator()(const SOM_element s) {
        int my_x;
        int my_y;
        double min_dist = my_map.BMU_range(s, my_x, my_y, my_square);
        ++fn_tally;  // count how many times this function_node executed
        return search_result_type(min_dist, my_x, my_y);
    }
};

typedef function_node<SOM_element, search_result_type> search_node;
typedef broadcast_node<SOM_element> b_node;
typedef std::vector< search_node *> search_node_vector_type;
typedef std::vector< search_node_vector_type > search_node_array_type;
typedef std::vector< graph *> graph_vector_type;
typedef std::vector< graph_vector_type > graph_array_type;

#define SPECULATION_CNT 2

graph *g[SPECULATION_CNT];  // main graph; there should only be one per epoch
b_node *send_to[SPECULATION_CNT];      // broadcast node to send exemplar to all function_nodes
queue_node<search_result_type> *q[SPECULATION_CNT];  // queue for function nodes to put their results in
// each function_node should have its own graph
search_node_array_type* s_array[SPECULATION_CNT];  // 2d array of function nodes
graph_array_type* g_array[SPECULATION_CNT];        // 2d array of graphs

// build a set of SPECULATION_CNT graphs, each of which consists of a broadcast_node,
//    xranges x yranges function_nodes, and one queue_node for output.
//    once speculation starts, if i % SPECULATION_CNT is the current graph, (i+1) % SPECULATION_CNT
//    is the first speculation, and so on.
void
build_BMU_graph(SOMap &map1) {
    // build current graph
    xsize = ((int)map1.size() + xranges - 1) / xranges;
    ysize = ((int)map1[0].size() + yranges - 1) / yranges;
    function_node_execs.clear();
    function_node_execs.reserve(xranges*yranges+1);
    for(int ii = 0; ii < xranges*yranges+1;++ii) function_node_execs.push_back(0);

    for(int scnt = 0; scnt < SPECULATION_CNT; ++scnt) {
        g[scnt] = new graph;
        send_to[scnt] = new b_node(*(g[scnt]));  // broadcast node to the function_nodes
        q[scnt] = new queue_node<search_result_type>(*(g[scnt]));  // output queue

        // create the function_nodes, tie to the graph
        s_array[scnt] = new search_node_array_type;
        s_array[scnt]->reserve(xranges);
        g_array[scnt] = new graph_array_type;
        g_array[scnt]->reserve(xranges);
        for(int i = 0; i < (int)map1.size(); i += xsize) {
            int xindex = i / xsize;
            s_array[scnt]->push_back(search_node_vector_type());
            (*s_array[scnt])[xindex].reserve(yranges);
            g_array[scnt]->push_back(graph_vector_type());
            (*g_array[scnt])[xindex].reserve(yranges);
            for( int j = 0; j < (int)map1[0].size(); j += ysize) {
                int offset = (i/xsize)*yranges + (j / ysize);
                int xmax = (i + xsize) > (int)map1.size() ? (int)map1.size() : i + xsize;
                int ymax = (j + ysize) > (int)map1[0].size() ? (int)map1[0].size() : j + ysize;
                subsquare_type sst(i,xmax,1,j,ymax,1);
                BMU_search_body bb(map1,sst,function_node_execs[offset]);
                graph *g_local = new graph;
                search_node *s = new search_node(*g_local, serial, bb); // copies Body
                (*g_array[scnt])[xindex].push_back(g_local);
                (*s_array[scnt])[xindex].push_back(s);
                make_edge(*(send_to[scnt]), *s);  // broadcast_node -> function_node
                make_edge(*s, *(q[scnt]));   // function_node -> queue_node
            }
        }
    }
}

// Wait for the 2D array of flow::graphs.
void wait_for_all_graphs(int cIndex) {  // cIndex ranges over [0 .. SPECULATION_CNT - 1]
    for(int x = 0; x < xranges; ++x) {
        for(int y = 0; y < yranges; ++y) {
            (*g_array[cIndex])[x][y]->wait_for_all();
        }
    }
}

void
destroy_BMU_graph() {
    for(int scnt = 0; scnt < SPECULATION_CNT; ++scnt) {
        for( int i = 0; i < (int)(*s_array[scnt]).size(); ++i ) {
            for(int j = 0; j < (int)(*s_array[scnt])[i].size(); ++j) {
                delete (*s_array[scnt])[i][j];
                delete (*g_array[scnt])[i][j];
            }
        }
        (*s_array[scnt]).clear();
        delete s_array[scnt];
        (*g_array[scnt]).clear();
        delete g_array[scnt];
        delete q[scnt];
        delete send_to[scnt];
        delete g[scnt];
    }
}

void find_subrange_overlap(int const &xval, int const &yval, double const &radius, int &xlow, int &xhigh, int &ylow, int &yhigh) {
    xlow = int((xval-radius)/xsize);
    xhigh = int((xval+radius)/xsize);
    ylow = int((yval-radius)/ysize);
    yhigh = int((yval+radius)/ysize);
    // circle may fall partly outside map
    if(xlow < 0) xlow = 0;
    if(xhigh >= xranges) xhigh = xranges - 1;
    if(ylow < 0) ylow = 0;
    if(yhigh >= yranges) yhigh = yranges - 1;
}

bool overlap( int &xval, int &yval, search_result_type &sr) {
    int xlow, xhigh, ylow, yhigh;
    find_subrange_overlap(get<XV>(sr), get<YV>(sr), get<RADIUS>(sr), xlow, xhigh, ylow, yhigh);
    return xval >= xlow && xval <= xhigh && yval >= ylow && yval <= yhigh;
}

void
cancel_submaps(int &xval, int &yval, double &radius, int indx) {
    int xlow;
    int xhigh;
    int ylow;
    int yhigh;
    find_subrange_overlap(xval, yval, radius, xlow, xhigh, ylow, yhigh);
    for(int x = xlow; x <= xhigh; ++x) {
        for(int y = ylow; y <= yhigh; ++y) {
            (*g_array[indx])[x][y]->root_task()->cancel_group_execution();
        }
    }
}

void
restart_submaps(int &xval, int &yval, double &radius, int indx, SOM_element &vector) {
    int xlow;
    int xhigh;
    int ylow;
    int yhigh;
    find_subrange_overlap(xval, yval, radius, xlow, xhigh, ylow, yhigh);
    for(int x = xlow; x <= xhigh; ++x) {
        for(int y = ylow; y <= yhigh; ++y) {
            // have to reset the graph
            (*g_array[indx])[x][y]->root_task()->context()->reset();
            // and re-submit the exemplar for search.
            (*s_array[indx])[x][y]->try_put(vector);
        }
    }
}

search_result_type
graph_BMU( int indx ) {  // indx ranges over [0 .. SPECULATION_CNT -1]
    wait_for_all_graphs(indx);  // wait for the array of subgraphs
    (g[indx])->wait_for_all();
    std::vector<search_result_type> all_srs(xRangeMax*yRangeMax,search_result_type(DBL_MAX,-1,-1));
    search_result_type sr;
    search_result_type min_sr;
    get<RADIUS>(min_sr) = DBL_MAX;
    int result_count = 0;
    while((q[indx])->try_get(sr)) {
        ++result_count;
        // figure which submap this came from
        int x = get<XV>(sr) / xsize;
        int y = get<YV>(sr) / ysize;
        int offset = x*yranges+y;  // linearized subscript
        all_srs[offset] = sr;
        if(get<RADIUS>(sr) < get<RADIUS>(min_sr))
            min_sr = sr;
        else if(get<RADIUS>(sr) == get<RADIUS>(min_sr)) {
            if(get<XV>(sr) < get<XV>(min_sr)) {
                min_sr = sr;
            }
            else if((get<XV>(sr) == get<XV>(min_sr) &&
                  get<YV>(sr) < get<YV>(min_sr)))
            {
                min_sr = sr;
            }
        }
    }
    return min_sr;
    // end of one epoch
}

void graph_teach(SOMap &map1, teaching_vector_type &in) {
    build_BMU_graph(map1);
    // normally the training would pick random exemplars to teach the SOM.  We need
    // the process to be reproducible, so we will pick the exemplars in order, [0, in.size())
    int next_j = 0;
    for(int epoch = 0; epoch < nPasses; ++epoch) {
        global_i = epoch;
        bool canceled_submaps = false;
        int j = next_j;  // try to make reproducible
        next_j = (epoch+1) % in.size();
        search_result_type min_sr;
        if(epoch < speculation_start) {
            (send_to[epoch%SPECULATION_CNT])->try_put(in[j]);
        }
        else if(epoch == speculation_start) {
            (send_to[epoch%SPECULATION_CNT])->try_put(in[j]);
            if(epoch < nPasses-1) {
                (send_to[(epoch+1)%SPECULATION_CNT])->try_put(in[next_j]);
            }
        }
        else if(epoch < nPasses - 1) {
            (send_to[(epoch+1)%SPECULATION_CNT])->try_put(in[next_j]);
        }
        min_sr = graph_BMU(epoch % SPECULATION_CNT);  //calls wait_for_all()
        double min_distance = get<0>(min_sr);
        double radius = max_radius * exp(-(double)epoch*radius_decay_rate);
        double learning_rate = max_learning_rate * exp(-(double)epoch * learning_decay_rate);
        if(epoch >= speculation_start && epoch < (nPasses - 1)) {
            // have to cancel the affected submaps
            cancel_submaps(get<XV>(min_sr), get<YV>(min_sr), radius, (epoch+1)%SPECULATION_CNT);
            canceled_submaps = true;
        }
        map1.epoch_update(in[j], epoch, get<1>(min_sr), get<2>(min_sr), radius, learning_rate);
        ++global_i;
        if(canceled_submaps) {
            // do I have to wait for all the non-canceled speculative graph to complete first?
            // yes, in case a canceled task was already executing.
            wait_for_all_graphs((epoch+1) % SPECULATION_CNT);  // wait for the array of subgraphs
            restart_submaps(get<1>(min_sr), get<2>(min_sr), radius, (epoch+1)%SPECULATION_CNT, in[next_j]);
        }

        last_update = min_sr;
        get<RADIUS>(last_update) = radius;  // not smallest value, but range of effect
    }
    destroy_BMU_graph();
}

static const double serial_time_adjust = 1.25;
static double radius_fraction = 3.0;

int
main(int argc, char** argv) {
    int l_speculation_start;
    utility::thread_number_range threads( 
            task_scheduler_init::default_num_threads,
            task_scheduler_init::default_num_threads()  // run only the default number of threads if none specified
    );

    utility::parse_cli_arguments(argc,argv,
            utility::cli_argument_pack()
            //"-h" option for for displaying help is present implicitly
            .positional_arg(threads,"n-of-threads","number of threads to use; a range of the form low[:high], where low and optional high are non-negative integers or 'auto' for the TBB default.")
            // .positional_arg(InputFileName,"input-file","input file name")
            // .positional_arg(OutputFileName,"output-file","output file name")
            .positional_arg(radius_fraction, "radius-fraction","size of radius at which to start speculating")
            .positional_arg(nPasses, "number-of-epochs","number of examples used in learning phase")
            .arg(cancel_test, "cancel-test", "test for cancel signal while finding BMU")
            .arg(extra_debug, "debug", "additional output")
            .arg(dont_speculate,"nospeculate","don't speculate in SOM map teaching")
         );

    readInputData();
    max_radius = (xMax < yMax) ? yMax / 2 : xMax / 2;
    // need this value for the 1x1 timing below
    radius_decay_rate = -(log(1.0/(double)max_radius) / (double)nPasses);
    find_data_ranges(my_teaching, max_range, min_range );
    if(extra_debug) {
        printf( "Data range: ");
        remark_SOM_element(min_range);
        printf( " to ");
        remark_SOM_element(max_range);
        printf( "\n");
    }

    // find how much time is taken for the single function_node case.
    // adjust nPasses so the 1x1 time is somewhere around serial_time_adjust seconds.
   // make sure the example test runs for at least 0.5 second.
    for(;;) {
        task_scheduler_init init(1);
        SOMap map1(xMax,yMax);
        speculation_start = nPasses + 1;  // Don't speculate

        xranges = 1;
        yranges = 1;
        map1.initialize(InitializeGradient, max_range, min_range);
        tick_count t0 = tick_count::now();
        graph_teach(map1, my_teaching);
        tick_count t1 = tick_count::now();
        double nSeconds = (t1-t0).seconds();
        if(nSeconds < 0.5) {
            xMax *= 2;
            yMax *= 2;
            continue;
        }
        double size_adjust = sqrt(serial_time_adjust / nSeconds);
        xMax = (int)((double)xMax * size_adjust);
        yMax = (int)((double)yMax * size_adjust);
        max_radius = (xMax < yMax) ? yMax / 2 : xMax / 2;
        radius_decay_rate = log((double)max_radius) / (double)nPasses;

        if(extra_debug) {
            printf("original 1x1 case ran in %g seconds\n", nSeconds);
            printf("   Size of table == %d x %d\n", xMax, yMax);
            printf("   radius_decay_rate == %g\n", radius_decay_rate);
        }
        break;
    }

    // the "max_radius" starts at 1/2*radius_fraction the table size.  To start the speculation when the radius is
    // 1 / n * the table size, the constant in the log below should be n / 2.  so 2 == 1/4, 3 == 1/6th,
    // et c.
    if(dont_speculate) {
        l_speculation_start = nPasses + 1;
        if ( extra_debug )printf("speculation will not be done\n");
    }
    else {
        if(radius_fraction < 1.0 ) {
            if ( extra_debug )printf("Warning: radius_fraction should be >= 1.  Setting to 1.\n");
            radius_fraction = 1.0;
        }
        l_speculation_start = (int)((double)nPasses * log(radius_fraction) / log((double)nPasses)); 
        if ( extra_debug )printf( "We will start speculation at iteration %d\n", l_speculation_start );
    }
    double single_time;  // for speedup calculations
    for(int p = threads.first; p <= threads.last; ++p) {
        task_scheduler_init init(p);
        if ( extra_debug )printf( " -------------- Running with %d threads. ------------\n", p);
       // run the SOM build for a series of subranges
        for(xranges = 1; xranges <= xRangeMax; ++xranges) {
            for(yranges = xranges; yranges <= yRangeMax; ++yranges) {
                if(xranges == 1 && yranges == 1) {
                    // don't pointlessly speculate if we're only running one subrange.
                    speculation_start = nPasses + 1;
                }
                else {
                    speculation_start = l_speculation_start;
                }
                SOMap map1(xMax, yMax);
                map1.initialize(InitializeGradient, max_range, min_range);
    
                if(extra_debug) printf( "Start learning for [%d,%d] ----------- \n", xranges,yranges);
                tick_count t0 = tick_count::now();
                graph_teach(map1, my_teaching);
                tick_count t1 = tick_count::now();
                
                if ( extra_debug )printf( "Done learning for [%d,%d], which took %g seconds ", xranges,yranges, (t1-t0).seconds());
                if(xranges == 1 && yranges == 1) single_time = (t1-t0).seconds();
                if ( extra_debug )printf( ": speedup == %g\n", single_time / (t1-t0).seconds());
    
            }  // yranges
        }  // xranges
    }  // #threads p
    printf("done\n");
    return 0;
}
