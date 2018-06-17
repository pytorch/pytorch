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

#ifndef FRACTAL_VIDEO_H_
#define FRACTAL_VIDEO_H_

#include "../../common/gui/video.h"
#include "fractal.h"

extern video *v;
extern bool single;

class fractal_video : public video
{
    fractal_group *fg;

private:
    void on_mouse( int x, int y, int key ) {
        if( key == 1 ) {
            if ( fg ) {
                fg->set_num_frames_at_least(20);
                fg->mouse_click( x, y );
            }
        }
    }

    void on_key( int key ) {
        switch ( key&0xff ) {
            case esc_key:
                running = false; break;
            case ' ': // space
                if( fg ) fg->switch_priorities(); break;

            case 'q':
                if( fg ) fg->active_fractal_zoom_in(); break;
            case 'e':
                if( fg ) fg->active_fractal_zoom_out(); break;

            case 'r':
                if( fg ) fg->active_fractal_quality_inc(); break;
            case 'f':
                if( fg ) fg->active_fractal_quality_dec(); break;

            case 'w':
                if( fg ) fg->active_fractal_move_up(); break;
            case 'a':
                if( fg ) fg->active_fractal_move_left(); break;
            case 's':
                if( fg ) fg->active_fractal_move_down(); break;
            case 'd':
                if( fg ) fg->active_fractal_move_right(); break;
        }
        if( fg ) fg->set_num_frames_at_least(20);
    }

    void on_process() {
        if ( fg ) {
            fg->run( !single );
        }
    }

public:
    fractal_video() :fg(0) {
        title = "Dynamic Priorities in TBB: Fractal Example";
        v = this;
    }

    void set_fractal_group( fractal_group &_fg ) {
        fg = &_fg;
    }
};

#endif /* FRACTAL_VIDEO_H_ */
