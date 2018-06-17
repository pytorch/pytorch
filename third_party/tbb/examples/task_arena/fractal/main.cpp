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

#define VIDEO_WINMAIN_ARGS

#include <stdio.h>
#include <iostream>

#include "fractal.h"
#include "fractal_video.h"

#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"

#include "../../common/utility/utility.h"

bool silent = false;
bool single = false;
bool schedule_auto = false;
int grain_size = 8;

int main(int argc, char *argv[])
{
    try{
        tbb::tick_count mainStartTime = tbb::tick_count::now();

        // It is used for console mode for test with different number of threads and also has
        // meaning for GUI: threads.first  - use separate event/updating loop thread (>0) or not (0).
        //                  threads.second - initialization value for scheduler
        utility::thread_number_range threads( tbb::task_scheduler_init::default_num_threads );
        int num_frames = -1;
        int max_iterations = 1000000;

        // command line parsing
        utility::parse_cli_arguments(argc,argv,
            utility::cli_argument_pack()
            //"-h" option for displaying help is present implicitly
            .positional_arg(threads,"n-of-threads",utility::thread_number_range_desc)
            .positional_arg(num_frames,"n-of-frames","number of frames the example processes internally")
            .positional_arg(max_iterations,"max-of-iterations","maximum number of the fractal iterations")
            .positional_arg(grain_size,"grain-size","the grain size value")
            .arg(schedule_auto, "use-auto-partitioner", "use tbb::auto_partitioner")
            .arg(silent, "silent", "no output except elapsed time")
            .arg(single, "single", "process only one fractal")
        );

        fractal_video video;

        // video layer init
        if ( video.init_window(1024, 512) ) {
            video.calc_fps = false;
            video.threaded = threads.first > 0;
            // initialize fractal group
            fractal_group fg( video.get_drawing_memory(), threads.last, max_iterations, num_frames );
            video.set_fractal_group( fg );
            // main loop
            video.main_loop();
        }
        else if ( video.init_console() ) {
            // in console mode we always have limited number of frames
            num_frames = num_frames<0 ? 1 : num_frames;
            for(int p = threads.first;  p <= threads.last; p = threads.step(p) ) {
                if ( !silent ) printf("Threads = %d\n", p);
                fractal_group fg( video.get_drawing_memory(), p, max_iterations, num_frames );
                fg.run( !single );
            }
        }
        video.terminate();
        utility::report_elapsed_time((tbb::tick_count::now() - mainStartTime).seconds());
        return 0;
    } catch ( std::exception& e ) {
        std::cerr<<"error occurred. error text is :\"" <<e.what()<<"\"\n";
        return 1;
    }
}
