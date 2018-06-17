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

#include "../../common/utility/utility.h"

#if __TBB_MIC_OFFLOAD
#pragma offload_attribute (push,target(mic))
#endif // __TBB_MIC_OFFLOAD

#include <cstdio>
#include <cstdlib>
#include <string>

#include "tbb/atomic.h"
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/task_group.h"

#pragma warning(disable: 4996)

const unsigned BOARD_SIZE=81;
const unsigned BOARD_DIM=9;

using namespace tbb;
using namespace std;

tbb::atomic<unsigned> nSols;
bool find_one = false;
bool verbose = false;
unsigned short init_values[BOARD_SIZE] = {1,0,0,9,0,0,0,8,0,0,8,0,2,0,0,0,0,0,0,0,5,0,0,0,7,0,0,0,5,2,1,0,0,4,0,0,0,0,0,0,0,5,0,0,7,4,0,0,7,0,0,0,3,0,0,3,0,0,0,2,0,0,5,0,0,0,0,0,0,1,0,0,5,0,0,0,1,0,0,0,0};
task_group *g;
double solve_time;

typedef struct {
    unsigned short solved_element;
    unsigned potential_set;
} board_element;

void read_board(const char *filename) {
    FILE *fp;
    int input;
    fp = fopen(filename, "r");
    if (!fp) { 
        fprintf(stderr, "sudoku: Could not open input file '%s'.\n", filename);
        exit(1);
    }
    for (unsigned i=0; i<BOARD_SIZE; ++i) {
        if (fscanf(fp, "%d", &input))
            init_values[i] = input;
        else {
            fprintf(stderr, "sudoku: Error in input file at entry %d, assuming 0.\n", i);
            init_values[i] = 0;
        }
    }
    fclose(fp);
}

void print_board(board_element *b) {
    for (unsigned row=0; row<BOARD_DIM; ++row) {
        for (unsigned col=0; col<BOARD_DIM; ++col) {
            printf(" %d", b[row*BOARD_DIM+col].solved_element);
            if (col==2 || col==5) printf(" |");
        }
        printf("\n");
        if (row==2 || row==5) printf(" ---------------------\n");
    }
}

void print_potential_board(board_element *b) {
    for (unsigned row=0; row<BOARD_DIM; ++row) {
        for (unsigned col=0; col<BOARD_DIM; ++col) {
            if (b[row*BOARD_DIM+col].solved_element) 
                printf("  %4d ", b[row*BOARD_DIM+col].solved_element);
            else
                printf(" [%4d]", b[row*BOARD_DIM+col].potential_set);
            if (col==2 || col==5) printf(" |");
        }
        printf("\n");
        if (row==2 || row==5)
            printf(" ------------------------------------------------------------------\n");
    }
}

void init_board(board_element *b) {
    for (unsigned i=0; i<BOARD_SIZE; ++i)
        b[i].solved_element = b[i].potential_set = 0;
}

void init_board(board_element *b, unsigned short arr[81]) {
    for (unsigned i=0; i<BOARD_SIZE; ++i) {
        b[i].solved_element = arr[i]; 
        b[i].potential_set = 0;
    }
}

void init_potentials(board_element *b) {
    for (unsigned i=0; i<BOARD_SIZE; ++i)
        b[i].potential_set = 0;
}

void copy_board(board_element *src, board_element *dst) {
    for (unsigned i=0; i<BOARD_SIZE; ++i)
        dst[i].solved_element = src[i].solved_element;
}

bool fixed_board(board_element *b) {
    for (int i=BOARD_SIZE-1; i>=0; --i)
        if (b[i].solved_element==0) return false;
    return true;
}

bool in_row(board_element *b, unsigned row, unsigned col, unsigned short p) {
    for (unsigned c=0; c<BOARD_DIM; ++c)
        if (c!=col && b[row*BOARD_DIM+c].solved_element==p)  return true;
    return false;
}

bool in_col(board_element *b, unsigned row, unsigned col, unsigned short p) {
    for (unsigned r=0; r<BOARD_DIM; ++r)
        if (r!=row && b[r*BOARD_DIM+col].solved_element==p)  return true;
    return false;
}

bool in_block(board_element *b, unsigned row, unsigned col, unsigned short p) {
    unsigned b_row = row/3 * 3, b_col = col/3 * 3;
    for (unsigned i=b_row; i<b_row+3; ++i)
        for (unsigned j=b_col; j<b_col+3; ++j)
            if (!(i==row && j==col) && b[i*BOARD_DIM+j].solved_element==p) return true;
    return false;
}

void calculate_potentials(board_element *b) {
    for (unsigned i=0; i<BOARD_SIZE; ++i) {
        b[i].potential_set = 0;
        if (!b[i].solved_element) { // element is not yet fixed
            unsigned row = i/BOARD_DIM, col = i%BOARD_DIM;
            for (unsigned potential=1; potential<=BOARD_DIM; ++potential) {
                if (!in_row(b, row, col, potential) && !in_col(b, row, col, potential)
                    && !in_block(b, row, col, potential))
                    b[i].potential_set |= 1<<(potential-1);
            }
        }
    }
}

bool valid_board(board_element *b) {
    bool success=true;
    for (unsigned i=0; i<BOARD_SIZE; ++i) {
        if (success && b[i].solved_element) { // element is fixed
            unsigned row = i/BOARD_DIM, col = i%BOARD_DIM;
            if (in_row(b, row, col, b[i].solved_element) || in_col(b, row, col, b[i].solved_element) || in_block(b, row, col, b[i].solved_element))
                success = false;
        }
    }
    return success;
}

bool examine_potentials(board_element *b, bool *progress) {
    bool singletons = false;
    for (unsigned i=0; i<BOARD_SIZE; ++i) {
        if (b[i].solved_element==0 && b[i].potential_set==0) // empty set
            return false;
        switch (b[i].potential_set) {
        case 1:   { b[i].solved_element = 1; singletons=true; break; }
        case 2:   { b[i].solved_element = 2; singletons=true; break; }
        case 4:   { b[i].solved_element = 3; singletons=true; break; }
        case 8:   { b[i].solved_element = 4; singletons=true; break; }
        case 16:  { b[i].solved_element = 5; singletons=true; break; }
        case 32:  { b[i].solved_element = 6; singletons=true; break; }
        case 64:  { b[i].solved_element = 7; singletons=true; break; }
        case 128: { b[i].solved_element = 8; singletons=true; break; }
        case 256: { b[i].solved_element = 9; singletons=true; break; }
        }
    }
    *progress = singletons;
    return valid_board(b);
}

#if !__TBB_CPP11_LAMBDAS_PRESENT
void partial_solve(board_element *b, unsigned first_potential_set);

class PartialSolveBoard {
    board_element *b;
    unsigned first_potential_set;
public:
    PartialSolveBoard(board_element *_b, unsigned fps) :
        b(_b), first_potential_set(fps) {}
    void operator() () const {
        partial_solve(b, first_potential_set);
    }
};
#endif

void partial_solve(board_element *b, unsigned first_potential_set) {
    if (fixed_board(b)) {
        if ( find_one )
            g->cancel();
        if (++nSols==1 && verbose) {
            print_board(b);
        }
        free(b);
        return;
    }
    calculate_potentials(b);
    bool progress=true;
    bool success = examine_potentials(b, &progress);
    if (success && progress) {
        partial_solve(b, first_potential_set);
    } else if (success && !progress) {
        board_element *new_board;
        while (b[first_potential_set].solved_element!=0) ++first_potential_set;
        for (unsigned short potential=1; potential<=BOARD_DIM; ++potential) {
            if (1<<(potential-1) & b[first_potential_set].potential_set) {
                new_board = (board_element *)malloc(BOARD_SIZE*sizeof(board_element));
                copy_board(b, new_board);
                new_board[first_potential_set].solved_element = potential;
#if __TBB_CPP11_LAMBDAS_PRESENT
                g->run( [=]{ partial_solve(new_board, first_potential_set); } );
#else
                g->run(PartialSolveBoard(new_board, first_potential_set));
#endif
            }
        }
        free(b);
    }
    else {
        free(b);
    }
}

unsigned solve(int p) {
    task_scheduler_init init(p);
    nSols = 0;
    board_element *start_board = (board_element *)malloc(BOARD_SIZE*sizeof(board_element));
    init_board(start_board, init_values);
    g = new task_group;
    tick_count t0 = tick_count::now();
    partial_solve(start_board, 0);
    g->wait();
    solve_time = (tick_count::now() - t0).seconds();
    delete g;
    return nSols;
}

#if __TBB_MIC_OFFLOAD
#pragma offload_attribute (pop)
#endif // __TBB_MIC_OFFLOAD

int do_get_default_num_threads() {
    int threads;
    #if __TBB_MIC_OFFLOAD
    #pragma offload target(mic) out(threads)
    #endif // __TBB_MIC_OFFLOAD
    threads = tbb::task_scheduler_init::default_num_threads();
    return threads;
}

int get_default_num_threads() {
    static int threads = do_get_default_num_threads();
    return threads;
}

int main(int argc, char *argv[]) {
    try {
        tbb::tick_count mainStartTime = tbb::tick_count::now();

        utility::thread_number_range threads(get_default_num_threads);
        string filename = "";
        bool silent = false;

        utility::parse_cli_arguments(argc,argv,
            utility::cli_argument_pack()
            //"-h" option for displaying help is present implicitly
            .positional_arg(threads,"n-of-threads",utility::thread_number_range_desc)
            .positional_arg(filename,"filename","input filename")

            .arg(verbose,"verbose","prints the first solution")
            .arg(silent,"silent","no output except elapsed time")
            .arg(find_one,"find-one","stops after finding first solution\n")
        );

        if ( silent ) verbose = false;

        if ( !filename.empty() )
            read_board( filename.c_str() );
        // otherwise (if file name not specified), the default statically initialized board will be used.
        for(int p = threads.first; p <= threads.last; p = threads.step(p) ) {
            unsigned number;
            #if __TBB_MIC_OFFLOAD
            #pragma offload target(mic) in(init_values, p, verbose, find_one) out(number, solve_time)
            {
            #endif // __TBB_MIC_OFFLOAD
            number = solve(p);
            #if __TBB_MIC_OFFLOAD
            }
            #endif // __TBB_MIC_OFFLOAD

            if ( !silent ) {
                if ( find_one ) {
                    printf("Sudoku: Time to find first solution on %d threads: %6.6f seconds.\n", p, solve_time);
                }
                else {
                    printf("Sudoku: Time to find all %u solutions on %d threads: %6.6f seconds.\n", number, p, solve_time);
                }
            }
        }

        utility::report_elapsed_time((tbb::tick_count::now() - mainStartTime).seconds());

        return 0;
    } catch(std::exception& e) {
        std::cerr<<"error occurred. error text is :\"" <<e.what()<<"\"\n";
        return 1;
    }
};

