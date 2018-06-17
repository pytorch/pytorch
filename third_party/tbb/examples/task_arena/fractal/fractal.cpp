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

#include "fractal.h"

#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/task_arena.h"
#include "tbb/task_group.h"
#include "tbb/tick_count.h"

#include <math.h>
#include <stdio.h>

// Included for __TBB_CPP11_LAMBDAS_PRESENT definition
#include "tbb/tbb_config.h"

video *v;
extern bool silent;
extern bool schedule_auto;
extern int grain_size;

color_t fractal::calc_one_pixel( int x0, int y0 ) const {
    unsigned int iter;
    double fx0, fy0, xtemp, x, y, mu;

    color_t color;

    fx0 = (double)x0 - (double) size_x / 2.0;
    fy0 = (double)y0 - (double) size_y / 2.0;
    fx0 = fx0 / magn + cx;
    fy0 = fy0 / magn + cy;

    iter = 0; x = 0; y = 0;
    mu = 0;

    while (((x*x + y*y) <= 4) && (iter < max_iterations)) {
        xtemp = x*x - y*y + fx0;
        y = 2*x*y + fy0;
        x = xtemp;
        mu += exp(-sqrt(x*x+y*y));
        iter++;
    }

    if (iter == max_iterations) {
        // point corresponds to the mandelbrot set
        color = v->get_color(255, 255, 255);
        return color;
    }

    int b = (int)(256*mu);
    int g = (b/8);
    int r = (g/16);

    b = b>255 ? 255 : b;
    g = g>255 ? 255 : g;
    r = r>255 ? 255 : r;

    color = v->get_color(r, g, b);
    return color;
}

void fractal::clear() {
    drawing_area area( off_x, off_y, size_x, size_y, dm ) ;

    // fill the rendering area with black color
    for (int y=0; y<size_y; ++y) {
        area.set_pos( 0, y );
        for (int x=0; x<size_x; ++x) {
            area.put_pixel( v->get_color(0, 0, 0) );
        }
    }
}

void fractal::draw_border( bool is_active ) {
    color_t color = is_active ? v->get_color(0, 255, 0) // green color
                                : v->get_color(96, 128, 96); // green-gray color

    // top border
    drawing_area area0( off_x-1, off_y-1, size_x+2, 1, dm );
    for (int i=-1; i<size_x+1; ++i)
        area0.put_pixel(color);
    // bottom border
    drawing_area area1( off_x-1, off_y+size_y, size_x+2, 1, dm );
    for (int i=-1; i<size_x+1; ++i)
        area1.put_pixel(color);
    // left border
    drawing_area area2( off_x-1, off_y, 1, size_y+2, dm );
    for (int i=0; i<size_y; ++i)
        area2.set_pixel(0, i, color);
    // right border
    drawing_area area3( size_x+off_x, off_y, 1, size_y+2, dm );
    for (int i=0; i<size_y; ++i)
        area3.set_pixel(0, i, color);
}

void fractal::render_rect( int x0, int y0, int x1, int y1 ) const {
    // render the specified rectangle area
    drawing_area area(off_x+x0, off_y+y0, x1-x0, y1-y0, dm);
    for ( int y=y0; y<y1; ++y ) {
        area.set_pos( 0, y-y0 );
        for ( int x=x0; x<x1; ++x ) {
            area.put_pixel( calc_one_pixel( x, y ) );
        }
    }
}

class fractal_body {
    fractal &f;
public:
    void operator()( tbb::blocked_range2d<int> &r ) const {
        if ( v->next_frame() )
            f.render_rect( r.cols().begin(), r.rows().begin(), r.cols().end(), r.rows().end() );
    }

    fractal_body( fractal &_f ) : f(_f) {
    }
};

void fractal::render( tbb::task_group_context &context ) {
    // Make copy of fractal object and render fractal with parallel_for with
    // the provided context and partitioner chosen by schedule_auto.
    // Updates to fractal are not reflected in the render.
    fractal f = *this;
    fractal_body body(f);

    if( schedule_auto )
        tbb::parallel_for( tbb::blocked_range2d<int>(0, size_y, grain_size, 0, size_x, grain_size ),
                body, tbb::auto_partitioner(), context);
    else
        tbb::parallel_for( tbb::blocked_range2d<int>(0, size_y, grain_size, 0, size_x, grain_size ),
                body, tbb::simple_partitioner(), context);
}

void fractal::run( tbb::task_group_context &context ) {
    clear();
    context.reset();
    render( context );
}

bool fractal::check_point( int x, int y ) const {
    return x >= off_x && x <= off_x+size_x &&
            y >= off_y && y <= off_y+size_y;
}

void fractal_group::calc_fractal( int num ) {
    // calculate the fractal
    fractal &f = num ? f1 : f0;

    tbb::tick_count t0 = tbb::tick_count::now();
    while ( v->next_frame() && num_frames[num] != 0 ) {
        f.run( context[num] );
        if ( num_frames[num]>0 ) num_frames[num] -= 1;
    }
    tbb::tick_count t1 = tbb::tick_count::now();

    if ( !silent ) {
        printf("  %s fractal finished. Time: %g\n", num ? "Second" : "First", (t1-t0).seconds());
    }
}

void fractal_group::set_priorities() {
    // set the high priority for the active area and the normal priority for another area
    context[active].set_priority( tbb::priority_high );
    context[active^1].set_priority( tbb::priority_low );
}

void fractal_group::switch_priorities( int new_active ) {
    if( new_active!=-1 ) active = new_active;
    else                 active = 1-active; // assumes 'active' is only 0 or 1
    set_priorities();
    draw_borders();
}

void fractal_group::set_num_frames_at_least( int n ) {
    if ( num_frames[0]<n ) num_frames[0] = n;
    if ( num_frames[1]<n ) num_frames[1] = n;
}

#if !__TBB_CPP11_LAMBDAS_PRESENT
class task_group_body {
    fractal_group &fg;
public:
    task_group_body(fractal_group &_fg) : fg(_fg) { }

    void operator() () const { fg.calc_fractal( 1 ); }
};

class arena_body {
    task_group_body &tg_body;
    tbb::task_group &task_group;
public:
    arena_body( task_group_body &_tg_body, tbb::task_group &_task_group )
        :  tg_body( _tg_body ), task_group( _task_group )  { }

    void operator() () const { task_group.run( tg_body ); }
};

class arena_body_wait {
    tbb::task_group &group;
public:
    arena_body_wait( tbb::task_group &gr ) : group(gr) { }

    void operator() () const { group.wait(); }
};
#endif

void fractal_group::run( bool create_second_fractal ) {
    // initialize task scheduler
    tbb::task_scheduler_init init( num_threads );

    // create contexts to manage fractal priorities
    context = new tbb::task_group_context[2];

    set_priorities();
    draw_borders();

    tbb::task_arena arena;
    tbb::task_group gr;

    // the second fractal is calculating on separated thread
    if ( create_second_fractal ) {
#if __TBB_CPP11_LAMBDAS_PRESENT
        arena.execute( [&] {
            gr.run( [&] { calc_fractal( 1 ); } );
        } );
#else
        task_group_body tg_body( *this );
        arena_body a_body( tg_body, gr );
        arena.execute( a_body );
#endif
    }

    // calculate the first fractal
    calc_fractal( 0 );

    if ( create_second_fractal ) {
#if __TBB_CPP11_LAMBDAS_PRESENT
        // wait for second fractal
        arena.execute( [&] { gr.wait(); } );
#else
        arena.execute( arena_body_wait( gr ) );
#endif
    }

    delete[] context;
}

void fractal_group::draw_borders() {
    f0.draw_border( active==0 );
    f1.draw_border( active==1 );
}

fractal_group::fractal_group( const drawing_memory &_dm, int _num_threads, unsigned int _max_iterations, int _num_frames ) : f0(_dm), f1(_dm), num_threads(_num_threads) {
    // set rendering areas
    f0.size_x = f1.size_x = _dm.sizex/2-4;
    f0.size_y = f1.size_y = _dm.sizey-4;
    f0.off_x = f0.off_y = f1.off_y = 2;
    f1.off_x = f0.size_x+4+2;

    // set fractals parameters
    f0.cx = -0.6f; f0.cy = 0.0f; f0.magn = 200.0f;
    f1.cx = -0.6f; f1.cy = 0.0f; f1.magn = 200.0f;
    f0.max_iterations = f1.max_iterations = _max_iterations;

    // initially the first fractal is active
    active = 0;

    num_frames[0] = num_frames[1] = _num_frames;
}

void fractal_group::mouse_click( int x, int y ) {
    // assumption that the point is not inside any fractal area
    int new_active = -1;

    if ( f0.check_point( x, y ) ) {
        // the point is inside the first fractal area
        new_active = 0;
    } else if ( f1.check_point( x, y ) ) {
        // the point is inside the second fractal area
        new_active = 1;
    }

    if ( new_active != -1 && new_active != active ) {
        switch_priorities( new_active );
    }
}
