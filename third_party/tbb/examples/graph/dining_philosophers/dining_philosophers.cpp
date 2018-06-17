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

#if _MSC_VER
    // Suppress "decorated name length exceeded, name was truncated" warning
    #pragma warning (disable: 4503)
#endif

#include "tbb/flow_graph.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"
#include "tbb/tbb_thread.h"
#include "tbb/atomic.h"
#include "tbb/spin_mutex.h"
#include <iostream>
#include "../../common/utility/utility.h"
#include <cstdlib>
#include <cstdio>

// Each philosopher is an object, and is invoked in the think() function_node, the
// eat() function_node and forward() multifunction_node.
//
// The graph is constructed, and each think() function_node is started with a continue_msg.
//
// The philosopher will think, then gather two chopsticks, eat, place the chopsticks back,
// and if they have not completed the required number of cycles, will start to think() again
// by sending a continue_msg to their corresponding think() function_node.
//
// The reserving join has as its inputs the left and right chopstick queues an a queue
// that stores the continue_msg emitted by the function_node after think()ing is done.
// When all three inputs are available, a tuple of the inputs will be forwarded to the
// eat() function_node.  The output of the eat() function_node is sent to the forward()
// multifunction_node.

const tbb::tick_count::interval_t think_time(1.0);
const tbb::tick_count::interval_t eat_time(1.0);
const int num_times = 10;

tbb::tick_count t0;
bool verbose = false;

const char *names[] = { "Archimedes", "Bakunin", "Confucius", "Democritus", "Euclid"
                      , "Favorinus", "Geminus", "Heraclitus", "Ichthyas", "Jason of Nysa",
                          "Kant", "Lavrov", "Metrocles", "Nausiphanes", "Onatas", "Phaedrus",
                          "Quillot", "Russell", "Socrates", "Thales", "Udayana",
                          "Vernadsky", "Wittgenstein", "Xenophilus", "Yen Yuan", "Zenodotus"
};
const int NumPhilosophers = sizeof(names) / sizeof(char*);

struct RunOptions {
    utility::thread_number_range threads;
    int number_of_philosophers;
    bool silent;
    RunOptions(utility::thread_number_range threads_, int number_of_philosophers_, bool silent_) :
        threads(threads_), number_of_philosophers(number_of_philosophers_), silent(silent_) { }
};

RunOptions ParseCommandLine(int argc, char *argv[]) {
    int auto_threads = tbb::task_scheduler_init::default_num_threads();
    utility::thread_number_range threads(tbb::task_scheduler_init::default_num_threads, auto_threads, auto_threads);
    int nPhilosophers = 5;
    bool verbose = false;
    char charbuf[100];
    std::sprintf(charbuf, "%d", NumPhilosophers);
    std::string pCount = "how many philosophers, from 2-";
    pCount += charbuf;

    utility::cli_argument_pack cli_pack;
    cli_pack.positional_arg(threads, "n-of_threads", utility::thread_number_range_desc)
            .positional_arg(nPhilosophers, "n-of-philosophers", pCount)
            .arg(verbose,"verbose","verbose output");
    utility::parse_cli_arguments(argc, argv, cli_pack);
    if(nPhilosophers < 2 || nPhilosophers > NumPhilosophers) {
        std::cout << "Number of philosophers (" << nPhilosophers << ") out of range [2:" << NumPhilosophers << "]\n";
        std::cout << cli_pack.usage_string(argv[0]) << std::flush;
        std::exit(1);
    }
    return RunOptions(threads, nPhilosophers,!verbose);
}


tbb::spin_mutex my_mutex;

class chopstick {};

using namespace tbb::flow;

typedef tbb::flow::tuple<continue_msg, chopstick, chopstick> join_output;
typedef join_node< join_output, reserving > join_node_type;

typedef function_node<continue_msg, continue_msg> think_node_type;
typedef function_node<join_output, continue_msg> eat_node_type;
typedef multifunction_node<continue_msg, join_output> forward_node_type;

class philosopher {
public:

    philosopher( const char *name ) :
        my_name(name), my_count(num_times) { }

    ~philosopher() {
    }

    void check();
    const char *name() const { return my_name; }

private:

    friend std::ostream& operator<<(std::ostream& o, philosopher const &p);

    const char *my_name;
    int my_count;

    friend class think_node_body;
    friend class eat_node_body;
    friend class forward_node_body;

    void think( );
    void eat();
    void forward( const continue_msg &in, forward_node_type::output_ports_type &out_ports );
};

std::ostream& operator<<(std::ostream& o, philosopher const &p) {
    o << "< philosopher[" << reinterpret_cast<uintptr_t>(const_cast<philosopher *>(&p)) << "] " << p.name()
        << ", my_count=" << p.my_count;
    return o;
}

class think_node_body {
    philosopher& my_philosopher;
public:
    think_node_body( philosopher &p ) : my_philosopher(p) { }
    think_node_body( const think_node_body &other ) : my_philosopher(other.my_philosopher) { }
    continue_msg operator()( continue_msg /*m*/) {
        my_philosopher.think();
        return continue_msg();
    }
};

class eat_node_body {
    philosopher &my_philosopher;
public:
    eat_node_body( philosopher &p) : my_philosopher(p) {}
    eat_node_body( const eat_node_body &other ) : my_philosopher(other.my_philosopher) { }
    continue_msg operator()(const join_output &in) {
        my_philosopher.eat();
        return continue_msg();
    }
};

class forward_node_body {
    philosopher &my_philosopher;
public:
    forward_node_body( philosopher &p) : my_philosopher(p) {}
    forward_node_body( const forward_node_body &other ) : my_philosopher(other.my_philosopher) { }
    void operator()( const continue_msg &in, forward_node_type::output_ports_type &out) {
        my_philosopher.forward( in, out);
    }
};

void philosopher::check() {
    if ( my_count != 0 ) {
        std::printf("ERROR: philosopher %s still had to run %d more times\n", name(), my_count);
        std::exit(1);
    }
}

void philosopher::forward( const continue_msg &/*in*/, forward_node_type::output_ports_type &out_ports ) {
    if(my_count < 0) abort();
    --my_count;
    (void)tbb::flow::get<1>(out_ports).try_put(chopstick());
    (void)tbb::flow::get<2>(out_ports).try_put(chopstick());
    if (my_count > 0) {
        (void)tbb::flow::get<0>(out_ports).try_put(continue_msg());  //start thinking again
    } else {
        if(verbose) {
            tbb::spin_mutex::scoped_lock lock(my_mutex);
            std::printf("%s has left the building\n", name());
        }
    }
}

void philosopher::eat() {
    if(verbose) {
        tbb::spin_mutex::scoped_lock lock(my_mutex);
        std::printf("%s eating\n", name());
    }
    tbb::this_tbb_thread::sleep(eat_time);
    if(verbose) {
        tbb::spin_mutex::scoped_lock lock(my_mutex);
        std::printf("%s done eating\n", name());
    }
}

void philosopher::think() {
    if(verbose) {
        tbb::spin_mutex::scoped_lock lock(my_mutex);
        std::printf("%s thinking\n", name());
    }
    tbb::this_tbb_thread::sleep(think_time);
    if(verbose) {
        tbb::spin_mutex::scoped_lock lock(my_mutex);
        std::printf("%s done thinking\n", name());
    }
}

typedef queue_node<continue_msg> thinking_done_type;

int main(int argc, char *argv[]) {
    try {
        tbb::tick_count main_time = tbb::tick_count::now();
        int num_threads;
        int num_philosophers;

        RunOptions options = ParseCommandLine(argc, argv);
        num_philosophers = options.number_of_philosophers;
        verbose = !options.silent;

    for(num_threads = options.threads.first; num_threads <= options.threads.last; num_threads = options.threads.step(num_threads)) {

        tbb::task_scheduler_init init(num_threads);

            graph g;

            if(verbose) std::cout << std::endl << num_philosophers << " philosophers with "
                << num_threads << " threads" << std::endl << std::endl;
            t0 = tbb::tick_count::now();

            std::vector<queue_node<chopstick> > places(num_philosophers, queue_node<chopstick>(g));
            std::vector<philosopher> philosophers;
            philosophers.reserve(num_philosophers);
            std::vector<think_node_type *> think_nodes;
            think_nodes.reserve(num_philosophers);
            std::vector<thinking_done_type> done_vector(num_philosophers, thinking_done_type(g));
            std::vector<join_node_type> join_vector(num_philosophers,join_node_type(g));
            std::vector<eat_node_type *> eat_nodes;
            eat_nodes.reserve(num_philosophers);
            std::vector<forward_node_type *> forward_nodes;
            forward_nodes.reserve(num_philosophers);
            for ( int i = 0; i < num_philosophers; ++i ) {
                places[i].try_put(chopstick());
                philosophers.push_back( philosopher( names[i] ) );  // allowed because of default generated assignment
                if(verbose) {
                    tbb::spin_mutex::scoped_lock lock(my_mutex);
                    std::cout << "Built philosopher " << philosophers[i] << std::endl;
                }
                think_nodes.push_back(new think_node_type(g, unlimited, think_node_body(philosophers[i])));
                eat_nodes.push_back( new eat_node_type(g, unlimited, eat_node_body(philosophers[i])));
                forward_nodes.push_back( new forward_node_type(g, unlimited, forward_node_body(philosophers[i])));
            }

            // attach chopstick buffers and think function_nodes to joins
            for(int i = 0; i < num_philosophers; ++i) {
                make_edge( *think_nodes[i], done_vector[i] );
                make_edge( done_vector[i], input_port<0>(join_vector[i]) );
                make_edge( places[i], input_port<1>(join_vector[i]) ); // left chopstick
                make_edge( places[(i+1) % num_philosophers], input_port<2>(join_vector[i]) ); // right chopstick
                make_edge( join_vector[i], *eat_nodes[i] );
                make_edge( *eat_nodes[i], *forward_nodes[i] );
                make_edge( output_port<0>(*forward_nodes[i]), *think_nodes[i] );
                make_edge( output_port<1>(*forward_nodes[i]), places[i] );
                make_edge( output_port<2>(*forward_nodes[i]), places[(i+1) % num_philosophers] );
            }

            // start all the philosophers thinking
            for(int i = 0; i < num_philosophers; ++i) think_nodes[i]->try_put(continue_msg());

            g.wait_for_all();

            tbb::tick_count t1 = tbb::tick_count::now();
            if(verbose) std::cout << std::endl << num_philosophers << " philosophers with "
                << num_threads << " threads have taken " << (t1-t0).seconds() << "seconds" << std::endl;

            for ( int i = 0; i < num_philosophers; ++i ) philosophers[i].check();

            for(int i = 0; i < num_philosophers; ++i) {
                delete think_nodes[i];
                delete eat_nodes[i];
                delete forward_nodes[i];
            }
        }

        utility::report_elapsed_time((tbb::tick_count::now() - main_time).seconds());
        return 0;
    } catch(std::exception& e) {
        std::cerr<<"error occurred. error text is :\"" <<e.what()<<"\"\n";
        return 1;
    }
}
