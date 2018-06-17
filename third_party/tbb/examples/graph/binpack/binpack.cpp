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


/* Bin-packing algorithm that attempts to use minimal number of bins B of
   size V to contain N items of varying sizes. */

#include <string>
#include <iostream>
#include <cmath>
#include <vector>
#include "tbb/atomic.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"
#include "tbb/flow_graph.h"
#include "../../common/utility/utility.h"

using tbb::tick_count;
using tbb::task_scheduler_init;
using namespace tbb::flow;

typedef size_t size_type;             // to represent non-zero indices, capacities, etc.
typedef size_t value_type;            // the type of items we are attempting to pack into bins
typedef std::vector<value_type> bin;  // we use a simple vector to represent a bin
// Our bin packers will be function nodes in the graph that take value_type items and
// return a dummy value.  They will also implicitly send packed bins to the bin_buffer
// node, and unused items back to the value_pool node:
typedef multifunction_node<value_type, tuple<value_type, bin>, rejecting> bin_packer;
// Items are placed into a pool that all bin packers grab from, represent by a queue_node:
typedef queue_node<value_type> value_pool;
// Packed bins are placed in this buffer waiting to be serially printed and/or accounted for:
typedef buffer_node<bin> bin_buffer;
// Packed bins are taken from the_bin_buffer and processed by the_writer:
typedef function_node<bin, continue_msg, rejecting> bin_writer;
// Items are injected into the graph when this node sends them to the_value_pool:
typedef source_node<value_type> value_source;

// User-specified globals with default values
size_type V = 42;                // desired capacity for each bin
size_type N = 1000;              // number of elements to generate
bool verbose = false;            // prints bin details and other diagnostics to screen
bool silent = false;             // suppress all output except for time
int num_bin_packers=-1;          // number of concurrent bin packers in operation; default is #threads;
                                 // larger values can result in more bins at less than full capacity
size_type optimality=1;          // 1 (default) is highest the algorithm can obtain; larger numbers run faster

// Calculated globals
size_type min_B;                 // lower bound on the optimal number of bins
size_type B;                     // the answer, i.e. number of bins used by the algorithm
size_type *input_array;          // stores randomly generated input values
value_type item_sum;             // sum of all randomly generated input values
tbb::atomic<value_type> packed_sum;   // sum of all values currently packed into all bins
tbb::atomic<size_type> packed_items;  // number of values currently packed into all bins
tbb::atomic<size_type> active_bins;   // number of active bin_packers
bin_packer **bins;               // the array of bin packers

// This class is the Body type for bin_packer
class bin_filler {
    typedef bin_packer::output_ports_type ports_type;
    bin my_bin;                   // the current bin that this bin_filler is packing
    size_type my_used;            // capacity of bin used by current contents (not to be confused with my_bin.size())
    size_type relax, relax_val;   // relaxation counter for determining when to settle for a non-full bin
    bin_packer* my_bin_packer;    // ptr to the bin packer that this body object is associated with
    size_type bin_index;          // index of the encapsulating bin packer in the global bins array
    value_type looking_for;       // the minimum size of item this bin_packer will accept
    value_pool* the_value_pool;   // the queue of incoming values
    bool done;                    // flag to indicate that this binpacker has been deactivated
 public:
    bin_filler(size_t bidx, value_pool* _q) :
        my_used(0), relax(0), relax_val(0), my_bin_packer(NULL), bin_index(bidx), looking_for(V), the_value_pool(_q), done(false) {}
    void operator()(const value_type& item, ports_type& p) {
        if (!my_bin_packer) my_bin_packer = bins[bin_index];
        if (done) get<0>(p).try_put(item); // this bin_packer is done packing items; put item back to pool
        else if (item > V) { // signal that packed_sum has reached item_sum at some point
            size_type remaining = active_bins--;
            if (remaining == 1 && packed_sum == item_sum) { // this is the last bin and it has seen everything
                // this bin_packer may not have seen everything, so stay active
                if (my_used>0) get<1>(p).try_put(my_bin);
                my_bin.clear();
                my_used = 0;
                looking_for = V;
                ++active_bins;
            }
            else if (remaining == 1) { // this is the last bin, but there are remaining items
                get<0>(p).try_put(V+1); // send out signal
                ++active_bins;
            }
            else if (remaining > 1) { // this is not the last bin; deactivate
                if (my_used < V/(1+optimality*.1)) { // this bin is ill-utilized; throw back items and deactivate
                    packed_sum -= my_used;
                    packed_items -= my_bin.size();
                    for (size_type i=0; i<my_bin.size(); ++i)
                        get<0>(p).try_put(my_bin[i]);
                    the_value_pool->remove_successor(*my_bin_packer); // deactivate
                    done = true;
                    get<0>(p).try_put(V+1); // send out signal
                }
                else { // this bin is well-utilized; send out bin and deactivate
                    the_value_pool->remove_successor(*my_bin_packer); // build no more bins
                    done = true;
                    if (my_used>0) get<1>(p).try_put(my_bin);
                    get<0>(p).try_put(V+1); // send out signal
                }
            }
        }
        else if (item <= V-my_used && item >= looking_for) { // this item can be packed
            my_bin.push_back(item);
            my_used += item;
            packed_sum += item;
            ++packed_items;
            looking_for = V-my_used;
            relax = 0;
            if (packed_sum == item_sum) {
                get<0>(p).try_put(V+1); // send out signal
            }
            if (my_used == V) {
                get<1>(p).try_put(my_bin);
                my_bin.clear();
                my_used = 0;
                looking_for = V;
            }
        }
        else { // this item can't be packed; relax constraints
            ++relax;
            if (relax >= (N-packed_items)/optimality) { // this bin_packer has looked through enough items
                relax = 0;
                --looking_for; // accept a wider range of items
                if (looking_for == 0 && my_used < V/(1+optimality*.1) && my_used > 0 && active_bins > 1) {
                    // this bin_packer is ill-utilized and can't find items; deactivate and throw back items
                    size_type remaining = active_bins--;
                    if (remaining > 1) { // not the last bin_packer
                        the_value_pool->remove_successor(*my_bin_packer); // deactivate
                        done = true;
                    }
                    else active_bins++; // can't deactivate last bin_packer
                    packed_sum -= my_used;
                    packed_items -= my_bin.size();
                    for (size_type i=0; i<my_bin.size(); ++i)
                        get<0>(p).try_put(my_bin[i]);
                    my_bin.clear();
                    my_used = 0;
                }
                else if (looking_for == 0 && (my_used >= V/(1+optimality*.1) || active_bins == 1)) {
                    // this bin_packer can't find items but is well-utilized, so send it out and reset
                    get<1>(p).try_put(my_bin);
                    my_bin.clear();
                    my_used = 0;
                    looking_for = V;
                }
            }
            get<0>(p).try_put(item); // put unused item back to pool
        }
    }
};

// source node uses this to send the values to the value_pool
class item_generator {
    size_type counter;
public:
    item_generator() : counter(0) {}
    bool operator()(value_type& m) {
        if (counter<N) {
            m = input_array[counter];
            ++counter;
            return true;
        }
        return false;
    }
};

// the terminal function_node uses this to gather stats and print bin information
class bin_printer {
    value_type running_count;
    size_type item_count;
    value_type my_min, my_max;
    double avg;
public:
    bin_printer() : running_count(0), item_count(0), my_min(V), my_max(0), avg(0) {}
    continue_msg operator()(bin b) {
        value_type sum=0;
        ++B;
        if (verbose)
            std::cout << "[ ";
        for (size_type i=0; i<b.size(); ++i) {
            if (verbose)
                std::cout << b[i] << " ";
            sum+=b[i];
            ++item_count;
        }
        if (sum < my_min) my_min = sum;
        if (sum > my_max) my_max = sum;
        avg += sum;
        running_count += sum;
        if (verbose)
        std::cout << "]=" << sum << "; Done/Packed/Total cap: " << running_count << "/" << packed_sum << "/" << item_sum
             << " items:" << item_count << "/" << packed_items << "/" << N << " B=" << B << std::endl;
        if (item_count == N) { // should be the last; print stats
            avg = avg/(double)B;
            if (!silent)
                std::cout << "SUMMARY: #Bins used: " << B << "; Avg size: " << avg << "; Max size: " << my_max
                     << "; Min size: " << my_min << "\n         Lower bound on optimal #bins: " << min_B
                     << "; Start #bins: " << num_bin_packers << std::endl;
        }
        return continue_msg();  // need to return something
    }
};

int get_default_num_threads() {
    static int threads = 0;
    if (threads == 0)
        threads = task_scheduler_init::default_num_threads();
    return threads;
}

int main(int argc, char *argv[]) {
    try {
        utility::thread_number_range threads(get_default_num_threads);
        utility::parse_cli_arguments(argc, argv,
                                     utility::cli_argument_pack()
                                     //"-h" option for displaying help is present implicitly
                                     .positional_arg(threads,"#threads",utility::thread_number_range_desc)
                                     .arg(verbose,"verbose","   print diagnostic output to screen")
                                     .arg(silent,"silent","    limits output to timing info; overrides verbose")
                                     .arg(N,"N","         number of values to pack")
                                     .arg(V,"V","         capacity of each bin")
                                     .arg(num_bin_packers,"#packers","  number of concurrent bin packers to use "
                                          "(default=#threads)")
                                     .arg(optimality,"optimality","controls optimality of solution; 1 is highest, use\n"
                                          "              larger numbers for less optimal but faster solution")
        );

        if (silent) verbose = false;  // make silent override verbose
        // Generate random input data
        srand(42);
        input_array = new value_type[N];
        item_sum = 0;
        for (size_type i=0; i<N; ++i) {
            input_array[i] = rand() % V + 1;  // generate items that fit in a bin
            item_sum += input_array[i];
        }
        min_B = (item_sum % V) ? item_sum/V + 1 : item_sum/V;

        tick_count start = tick_count::now();
        for(int p = threads.first; p <= threads.last; p = threads.step(p)) {
            task_scheduler_init init(p);
            packed_sum = 0;
            packed_items = 0;
            B = 0;
            if (num_bin_packers == -1) num_bin_packers = p;
            active_bins = num_bin_packers;
            if (!silent)
                std::cout << "binpack running with " << item_sum << " capacity over " << N << " items, optimality="
                     << optimality << ", " << num_bin_packers << " bins of capacity=" << V << " on " << p
                     << " threads.\n";
            graph g;
            value_source the_source(g, item_generator(), false);
            value_pool the_value_pool(g);
            make_edge(the_source, the_value_pool);
            bin_buffer the_bin_buffer(g);
            bins = new bin_packer*[num_bin_packers];
            for (int i=0; i<num_bin_packers; ++i) {
                bins[i] = new bin_packer(g, 1, bin_filler(i, &the_value_pool));
                make_edge(the_value_pool, *(bins[i]));
                make_edge(output_port<0>(*(bins[i])), the_value_pool);
                make_edge(output_port<1>(*(bins[i])), the_bin_buffer);
            }
            bin_writer the_writer(g, 1, bin_printer());
            make_edge(the_bin_buffer, the_writer);
            the_source.activate();
            g.wait_for_all();
            for (int i=0; i<num_bin_packers; ++i) {
                delete bins[i];
            }
            delete[] bins;
        }
        utility::report_elapsed_time((tick_count::now() - start).seconds());
        delete[] input_array;
        return 0;
    } catch(std::exception& e) {
        std::cerr<<"error occurred. error text is :\"" <<e.what()<<"\"\n";
        return 1;
    }
}
