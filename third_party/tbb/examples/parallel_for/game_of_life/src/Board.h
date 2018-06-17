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

#ifndef __BOARD_H__ 
#define __BOARD_H__

#define WIN32_LEAN_AND_MEAN

#ifndef _CONSOLE
#include <windows.h>

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;
#define LabelPtr Label^
#define BoardPtr Board^
#else
#define LabelPtr int*
#define BoardPtr Board*
#endif

struct Matrix 
{
    int width;
    int height;
    char* data;
};

#ifndef _CONSOLE
public ref class Board : public System::Windows::Forms::UserControl
#else
class Board
#endif
    {
    public:
        Board(int width, int height, int squareSize, LabelPtr counter);        
        virtual ~Board();
        void seed(int s);
        void seed(const BoardPtr s);
#ifndef _CONSOLE
    protected: 
        virtual void OnPaint(PaintEventArgs^ e) override;        
        void Board::draw(Graphics^ g);

    private:
        System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
        void InitializeComponent(void)
        {
            this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
        }
#pragma endregion

    private: delegate void drawDelegate(Int32);
    public:
        //! Called from the Evolution thread
        void draw( Int32 nCurIteration )
        {
            if (this->InvokeRequired)
            {
                drawDelegate^ d = gcnew drawDelegate(this, &Board::draw);
                IAsyncResult^ result = BeginInvoke(d, nCurIteration);
                EndInvoke(result);
                return;
            }
            m_counter->Text = nCurIteration.ToString();
            Invalidate();
        }
#endif
    public:
        Matrix *m_matrix;    

    private:
#ifndef _CONSOLE
        SolidBrush^ m_occupiedBrush;
        SolidBrush^ m_freeBrush;
        Graphics^ m_graphics;
        Graphics^ m_mem_dc;
        Bitmap^ m_bmp;
#endif
        int m_width;
        int m_height;
        int m_squareSize;
        LabelPtr m_counter;
    };
#endif
