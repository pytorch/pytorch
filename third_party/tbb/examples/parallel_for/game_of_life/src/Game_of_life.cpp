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
    Game_of_life.cpp : 
                      main project file.
*/
#include "Board.h"
#include "Form1.h"

#define WIN32_LEAN_AND_MEAN

#ifndef _CONSOLE
#include <windows.h>
#else
#include <iostream>
#include <sstream>
#include <time.h>
#include "Evolution.h"

#define BOARD_SQUARE_SIZE 2

int low;                            //! lower range limit of threads
int high;                           //! high range limit of threads
double execution_time;              //! time for game of life iterations
#endif

Board::Board(int width, int height, int squareSize, LabelPtr counter)
: m_width(width), m_height(height), m_squareSize(squareSize), m_counter(counter)
{
#ifndef _CONSOLE
    InitializeComponent();
    DoubleBuffered = true;

    this->Width = m_squareSize*width;
    this->Height = m_squareSize*height;
#endif
    m_matrix = new Matrix();
    m_matrix->width = width;
    m_matrix->height = height;
    m_matrix->data = new char[width*height];
    memset(m_matrix->data, 0, width*height);
#ifndef _CONSOLE
    m_occupiedBrush = gcnew SolidBrush(Color::Black);
    m_freeBrush = gcnew SolidBrush(Color::LightGray);
    
    m_graphics = CreateGraphics();
    m_bmp = gcnew Bitmap(Width, Height);
    m_mem_dc = Graphics::FromImage(m_bmp);
#endif
}

Board::~Board()
{
#ifndef _CONSOLE
    if (components)
    {
        delete components;
    }
#endif
    delete[] m_matrix->data;
    delete m_matrix;
}

void Board::seed(int s)
{        
    srand(s);
    for (int j=0; j<m_height; j++)
    {
        for (int i=0; i<m_width; i++)
        {        
            int x = rand()/(int)(((unsigned)RAND_MAX + 1) / 100);
            m_matrix->data[i+j*m_width] = x>75? 1: 0;               // 25% occupied
        }
    }
#ifndef _CONSOLE
    Invalidate();
#endif
}

void Board::seed( const BoardPtr src )
{        
            memcpy(m_matrix->data, src->m_matrix->data, m_height*m_width);
#ifndef _CONSOLE
    Invalidate();
#endif
}

#ifndef _CONSOLE
void Board::draw(Graphics^ g)
{
    m_mem_dc->FillRectangle(m_freeBrush, Drawing::Rectangle(0, 0, m_width*m_squareSize, m_height*m_squareSize));
    for (int j=0; j<m_height; j++)
    {
        for (int i=0; i<m_width; i++)
        {    
            if ( m_matrix->data[i+j*m_width] )
            {
                m_mem_dc->FillRectangle(m_occupiedBrush, Drawing::Rectangle(i*m_squareSize, j*m_squareSize, m_squareSize, m_squareSize));
            }
        }
    }
    g->DrawImage(m_bmp, 0, 0);
}

void Board::OnPaint(PaintEventArgs^ e)
{
    draw(e->Graphics);
}

[STAThreadAttribute]
int main(array<System::String ^> ^args)
{
    // Enabling Windows XP visual effects before any controls are created
    Application::EnableVisualStyles();
    Application::SetCompatibleTextRenderingDefault(false); 

    // Create the main window and run it
    Application::Run(gcnew Form1());
    return 0;
}
#else

//! Print usage of this program
void PrintUsage() 
{
    printf("Usage: gol [M[:N] -t execution_time]\nM and N are a range of numbers of threads to be used.\nexecution_time is a time (in sec) for execution game_of_life iterations\n");
    printf("Default values:\nM:\t\tautomatic\nN:\t\tM\nexecution_time:\t10\n");
}

//! Parse command line
bool ParseCommandLine(int argc, char * argv []) 
{
    char* s = argv[1];
    char* end;
    //! command line without parameters
    if(argc == 1)
    {
        low = tbb::task_scheduler_init::automatic;
        high = low;
        execution_time = 5;
        return true;
    }
    //! command line with parameters
    if(argc != 4)
    {
        PrintUsage();
        return false;
    }
    if(std::string("-t") != argv[argc-2])
    //! process M[:N] parameter
    high = strtol(s,&end,0);
    low = strtol(s,&end,0);
    switch( *end ) 
    {
        case ':': 
            high = strtol(end+1,0,0); 
            break;
        case '\0':
            break;
        default:
            PrintUsage();
            return false;
    }
    if (high < low)
    {
        std::cout << "Set correct range. Current range: " << low << ":" << high << std::endl;
        PrintUsage();
        return false;

    }
    //! process execution_time parameter
    execution_time = strtol(argv[argc-1],&end,0);
    return true;
}

int main( int argc, char* argv[] ) 
{
    if(!ParseCommandLine( argc, argv ))
        return 1;
    SequentialEvolution* m_seq;
    ParallelEvolution* m_par;
    Board* m_board1;
    Board* m_board2; 
    int* count = NULL;
   
    int boardWidth = 300;
    int boardHeight = 300;

    m_board1 = new Board(boardWidth, boardHeight, BOARD_SQUARE_SIZE, count);
    m_board2 = new Board(boardWidth, boardHeight, BOARD_SQUARE_SIZE, count);

    time_t now = time(NULL);
    printf("Generate Game of life board\n");
    m_board1->seed((int)now);
    m_board2->seed(m_board1);

    m_seq = new SequentialEvolution(m_board1->m_matrix, m_board1);
    m_seq->Run(execution_time, 1);
    delete m_seq;

    m_par = new ParallelEvolution(m_board2->m_matrix, m_board2);
    for( int p = low; p <= high; ++p ) 
    {
        m_par->Run(execution_time, p);
    }
    delete m_par;

    delete m_board1;
    delete m_board2;
    return 0;
}
#endif
