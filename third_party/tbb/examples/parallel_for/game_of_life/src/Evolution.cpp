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
    Evolution.cpp: implementation file for evolution classes; evolution 
                  classes do looped evolution of patterns in a defined 
                  2 dimensional space
*/

#include "Evolution.h"
#include "Board.h"

#ifdef USE_SSE
#define GRAIN_SIZE 14
#else
#define GRAIN_SIZE 4000
#endif
#define TIME_SLICE 330

/*
    Evolution
*/

/**
    Evolution::UpdateMatrix() - moves the calculated destination data 
    to the source data block. No destination zeroing is required since it will 
    be completely overwritten during the next calculation cycle.
**/
void Evolution::UpdateMatrix()
{
    memcpy(m_matrix->data, m_dest, m_size);
}

/*
    SequentialEvolution
*/

//! SequentialEvolution::Run - begins looped evolution
#ifndef _CONSOLE
void SequentialEvolution::Run()
{
#else
void SequentialEvolution::Run(double execution_time, int nthread)
{
    printf("Starting game (Sequential evolution)\n");
#endif

    m_nIteration = 0;
    m_serial_time = 0;
    tbb::tick_count t0 = tbb::tick_count::now();
    while (!m_done)
    {
        if( !is_paused ) 
        {
            tbb::tick_count t = tbb::tick_count::now();
            Step();
            tbb::tick_count t1 = tbb::tick_count::now();
            ++m_nIteration;
            double  work_time = (t1-t0).seconds();
#ifndef _CONSOLE
            if ( work_time * 1000 < TIME_SLICE )
                continue;
            m_serial_time += work_time;
            m_board->draw(m_nIteration);
#else
            m_serial_time += work_time;
#endif
        }
        //! Let the parallel algorithm work uncontended almost the same time
        //! as the serial one. See ParallelEvolution::Run() as well.
#ifndef _CONSOLE
        m_evt_start_parallel->Set();
        m_evt_start_serial->WaitOne();
        t0 = tbb::tick_count::now();
#else
        t0 = tbb::tick_count::now();
        if(m_serial_time > execution_time)
        {
            printf("iterations count = %d time = %g\n", m_nIteration, m_serial_time);
            break;
        }
#endif
    }
}

//! SequentialEvolution::Step() - override of step method
void SequentialEvolution::Step()
{
        if( !is_paused ) 
    {
#ifdef USE_SSE
    UpdateState(m_matrix, m_matrix->data, 0, m_matrix->height);
#else
    UpdateState(m_matrix, m_dest, 0, (m_matrix->width * m_matrix->height)-1);
    UpdateMatrix();
#endif
        }
}

/*
    ParallelEvolution
*/

//! SequentialEvolution::Run - begins looped evolution
#ifndef _CONSOLE
void ParallelEvolution::Run()
{
#else
void ParallelEvolution::Run(double execution_time, int nthread)
{
    if(nthread == tbb::task_scheduler_init::automatic)
        printf("Starting game (Parallel evolution for automatic number of thread(s))\n");
    else
        printf("Starting game (Parallel evolution for %d thread(s))\n", nthread);
#endif

    m_nIteration = 0;
    m_parallel_time = 0;

#ifndef _CONSOLE
    //! start task scheduler as necessary
    if (m_pInit == NULL)
    {
        m_pInit = new tbb::task_scheduler_init();
    }
    m_evt_start_parallel->WaitOne();
#else
    tbb::task_scheduler_init init(nthread);
#endif

    double  work_time = m_serial_time;
    tbb::tick_count t0 = tbb::tick_count::now();

    while (!m_done)
    {
        if( !is_paused ) 
        {
            tbb::tick_count t = tbb::tick_count::now();
            Step();
            tbb::tick_count t1 = tbb::tick_count::now();
            ++m_nIteration;
            double real_work_time = (t1-t0).seconds();
#ifndef _CONSOLE
            if ( real_work_time < work_time )
                continue;
            m_parallel_time += real_work_time;
            m_board->draw(m_nIteration); 
#else
            m_parallel_time += real_work_time;
#endif
        }
        //! Let the serial algorithm work the same time as the parallel one.
#ifndef _CONSOLE
        m_evt_start_serial->Set();
        m_evt_start_parallel->WaitOne();

        work_time = m_serial_time - m_parallel_time;
        t0 = tbb::tick_count::now();
#else
        t0 = tbb::tick_count::now();
        if(m_parallel_time > execution_time)
        {
            printf("iterations count = %d time = %g\n", m_nIteration, m_parallel_time);
            init.terminate();
            break;
        }
#endif
    }
}

/**
    class tbb_parallel_task
    
    TBB requires a class for parallel loop implementations. The actual 
    loop "chunks" are performed using the () operator of the class. 
    The blocked_range contains the range to calculate. Please see the 
    TBB documentation for more information.
**/
#ifndef _CONSOLE
public class tbb_parallel_task
#else
class tbb_parallel_task
#endif
{
public:
    static void set_values (Matrix* source, char* dest)
    {
        m_source = source;
        m_dest = dest;
        return;
    }

    void operator()( const tbb::blocked_range<size_t>& r ) const 
    {
        int begin = (int)r.begin();            //! capture lower range number for this chunk
        int end = (int)r.end();                //! capture upper range number for this chunk
        UpdateState(m_source, m_dest, begin, end);
    }

    tbb_parallel_task () {}

private:
    static Matrix* m_source;
    static char* m_dest;
};

Matrix* tbb_parallel_task::m_source;
char* tbb_parallel_task::m_dest;

//! ParallelEvolution::Step() - override of Step method
void ParallelEvolution::Step()
{
    size_t begin = 0;                   //! beginning cell position
#ifdef USE_SSE
    size_t end = m_matrix->height;      //! ending cell position
#else
    size_t end = m_size-1;              //! ending cell position
#endif

    //! set matrix pointers
    tbb_parallel_task::set_values(m_matrix, m_dest);

    //! do calculation loop
    parallel_for (tbb::blocked_range<size_t> (begin, end, GRAIN_SIZE), tbb_parallel_task());
    UpdateMatrix();
}
