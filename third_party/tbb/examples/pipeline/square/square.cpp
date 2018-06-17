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
// Example program that reads a file of decimal integers in text format
// and changes each to its square.
// 
#include "tbb/pipeline.h"
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tbb_allocator.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cctype>
#include "../../common/utility/utility.h"

extern void generate_if_needed(const char*);

using namespace std;

//! Holds a slice of text.
/** Instances *must* be allocated/freed using methods herein, because the C++ declaration
    represents only the header of a much larger object in memory. */
class TextSlice {
    //! Pointer to one past last character in sequence
    char* logical_end;
    //! Pointer to one past last available byte in sequence.
    char* physical_end;
public:
    //! Allocate a TextSlice object that can hold up to max_size characters.
    static TextSlice* allocate( size_t max_size ) {
        // +1 leaves room for a terminating null character.
        TextSlice* t = (TextSlice*)tbb::tbb_allocator<char>().allocate( sizeof(TextSlice)+max_size+1 );
        t->logical_end = t->begin();
        t->physical_end = t->begin()+max_size;
        return t;
    }
    //! Free a TextSlice object 
    void free() {
        tbb::tbb_allocator<char>().deallocate((char*)this,sizeof(TextSlice)+(physical_end-begin())+1);
    } 
    //! Pointer to beginning of sequence
    char* begin() {return (char*)(this+1);}
    //! Pointer to one past last character in sequence
    char* end() {return logical_end;}
    //! Length of sequence
    size_t size() const {return logical_end-(char*)(this+1);}
    //! Maximum number of characters that can be appended to sequence
    size_t avail() const {return physical_end-logical_end;}
    //! Append sequence [first,last) to this sequence.
    void append( char* first, char* last ) {
        memcpy( logical_end, first, last-first );
        logical_end += last-first;
    }
    //! Set end() to given value.
    void set_end( char* p ) {logical_end=p;}
};

size_t MAX_CHAR_PER_INPUT_SLICE = 4000;
string InputFileName = "input.txt";
string OutputFileName = "output.txt";

class MyInputFilter: public tbb::filter {
public:
    MyInputFilter( FILE* input_file_ );
    ~MyInputFilter();
private:
    FILE* input_file;
    TextSlice* next_slice;
    void* operator()(void*) /*override*/;
};

MyInputFilter::MyInputFilter( FILE* input_file_ ) : 
    filter(serial_in_order),
    input_file(input_file_),
    next_slice( TextSlice::allocate( MAX_CHAR_PER_INPUT_SLICE ) )
{ 
}

MyInputFilter::~MyInputFilter() {
    next_slice->free();
}
 
void* MyInputFilter::operator()(void*) {
    // Read characters into space that is available in the next slice.
    size_t m = next_slice->avail();
    size_t n = fread( next_slice->end(), 1, m, input_file );
    if( !n && next_slice->size()==0 ) {
        // No more characters to process
        return NULL;
    } else {
        // Have more characters to process.
        TextSlice& t = *next_slice;
        next_slice = TextSlice::allocate( MAX_CHAR_PER_INPUT_SLICE );
        char* p = t.end()+n;
        if( n==m ) {
            // Might have read partial number.  If so, transfer characters of partial number to next slice.
            while( p>t.begin() && isdigit(p[-1]) ) 
                --p;
            next_slice->append( p, t.end()+n );
        }
        t.set_end(p);
        return &t;
    }
}
    
//! Filter that changes each decimal number to its square.
class MyTransformFilter: public tbb::filter {
public:
    MyTransformFilter();
    void* operator()( void* item ) /*override*/;
};

MyTransformFilter::MyTransformFilter() : 
    tbb::filter(parallel) 
{}  

void* MyTransformFilter::operator()( void* item ) {
    TextSlice& input = *static_cast<TextSlice*>(item);
    // Add terminating null so that strtol works right even if number is at end of the input.
    *input.end() = '\0';
    char* p = input.begin();
    TextSlice& out = *TextSlice::allocate( 2*MAX_CHAR_PER_INPUT_SLICE );
    char* q = out.begin();
    for(;;) {
        while( p<input.end() && !isdigit(*p) ) 
            *q++ = *p++; 
        if( p==input.end() ) 
            break;
        long x = strtol( p, &p, 10 );
        // Note: no overflow checking is needed here, as we have twice the 
        // input string length, but the square of a non-negative integer n 
        // cannot have more than twice as many digits as n.
        long y = x*x; 
        sprintf(q,"%ld",y);
        q = strchr(q,0);
    }
    out.set_end(q);
    input.free();
    return &out;
}
         
//! Filter that writes each buffer to a file.
class MyOutputFilter: public tbb::filter {
    FILE* my_output_file;
public:
    MyOutputFilter( FILE* output_file );
    void* operator()( void* item ) /*override*/;
};

MyOutputFilter::MyOutputFilter( FILE* output_file ) : 
    tbb::filter(serial_in_order),
    my_output_file(output_file)
{
}

void* MyOutputFilter::operator()( void* item ) {
    TextSlice& out = *static_cast<TextSlice*>(item);
    size_t n = fwrite( out.begin(), 1, out.size(), my_output_file );
    if( n!=out.size() ) {
        fprintf(stderr,"Can't write into file '%s'\n", OutputFileName.c_str());
        exit(1);
    }
    out.free();
    return NULL;
}

bool silent = false;

int run_pipeline( int nthreads )
{
    FILE* input_file = fopen( InputFileName.c_str(), "r" );
    if( !input_file ) {
        throw std::invalid_argument( ("Invalid input file name: "+InputFileName).c_str() );
        return 0;
    }
    FILE* output_file = fopen( OutputFileName.c_str(), "w" );
    if( !output_file ) {
        throw std::invalid_argument( ("Invalid output file name: "+OutputFileName).c_str() );
        return 0;
    }

    // Create the pipeline
    tbb::pipeline pipeline;

    // Create file-reading writing stage and add it to the pipeline
    MyInputFilter input_filter( input_file );
    pipeline.add_filter( input_filter );

    // Create squaring stage and add it to the pipeline
    MyTransformFilter transform_filter; 
    pipeline.add_filter( transform_filter );

    // Create file-writing stage and add it to the pipeline
    MyOutputFilter output_filter( output_file );
    pipeline.add_filter( output_filter );

    // Run the pipeline
    tbb::tick_count t0 = tbb::tick_count::now();
    // Need more than one token in flight per thread to keep all threads 
    // busy; 2-4 works
    pipeline.run( nthreads*4 );
    tbb::tick_count t1 = tbb::tick_count::now();

    fclose( output_file );
    fclose( input_file );

    if ( !silent ) printf("time = %g\n", (t1-t0).seconds());

    return 1;
}

int main( int argc, char* argv[] ) {
    try {
        tbb::tick_count mainStartTime = tbb::tick_count::now();

        // The 1st argument is the function to obtain 'auto' value; the 2nd is the default value
        // The example interprets 0 threads as "run serially, then fully subscribed"
        utility::thread_number_range threads( tbb::task_scheduler_init::default_num_threads, 0 );

        utility::parse_cli_arguments(argc,argv,
            utility::cli_argument_pack()
            //"-h" option for displaying help is present implicitly
            .positional_arg(threads,"n-of-threads",utility::thread_number_range_desc)
            .positional_arg(InputFileName,"input-file","input file name")
            .positional_arg(OutputFileName,"output-file","output file name")
            .positional_arg(MAX_CHAR_PER_INPUT_SLICE, "max-slice-size","the maximum number of characters in one slice")
            .arg(silent,"silent","no output except elapsed time")
            );
        generate_if_needed( InputFileName.c_str() );

        if ( threads.first ) {
            for(int p = threads.first;  p <= threads.last; p=threads.step(p) ) {
                if ( !silent ) printf("threads = %d ", p);
                tbb::task_scheduler_init init(p);
                if(!run_pipeline (p))
                    return 1;
            }
        } else { // Number of threads wasn't set explicitly. Run serial and parallel version
            { // serial run
                if ( !silent ) printf("serial run   ");
                tbb::task_scheduler_init init_serial(1);
                if(!run_pipeline (1))
                    return 1;
            }
            { // parallel run (number of threads is selected automatically)
                if ( !silent ) printf("parallel run ");
                tbb::task_scheduler_init init_parallel;
                if(!run_pipeline (init_parallel.default_num_threads()))
                    return 1;
            }
        }

        utility::report_elapsed_time((tbb::tick_count::now() - mainStartTime).seconds());

        return 0;
    } catch(std::exception& e) {
        std::cerr<<"error occurred. error text is :\"" <<e.what()<<"\"\n";
        return 1;
    }
}
