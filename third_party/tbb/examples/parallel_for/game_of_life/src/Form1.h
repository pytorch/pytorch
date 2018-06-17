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

#ifndef _CONSOLE
#ifndef __FORM1_H__
#define __FORM1_H__

#include <time.h>
#include "Board.h"
#include "Evolution.h"

#define BOARD_SQUARE_SIZE 2

    using namespace System;
    using namespace System::ComponentModel;
    using namespace System::Collections;
    using namespace System::Windows::Forms;
    using namespace System::Data;
    using namespace System::Drawing;

    public ref class Form1 : public System::Windows::Forms::Form
    {
    public:
        Form1(void)
        {
            InitializeComponent();

            FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
            ClientSize = System::Drawing::Size(1206, 600+m_ribbonHeight+menuStrip1->Height);

            int boardWidth = (ClientRectangle.Width/2-m_sepWidth/2)/BOARD_SQUARE_SIZE;
            int boardHeight = (ClientRectangle.Height-menuStrip1->Height-m_ribbonHeight)/BOARD_SQUARE_SIZE;

            m_board1 = gcnew Board(boardWidth, boardHeight, BOARD_SQUARE_SIZE, seqGen);
            m_board2 = gcnew Board(boardWidth, boardHeight, BOARD_SQUARE_SIZE, parGen);
            
            Controls->Add(m_board1);
            Controls->Add(m_board2);

            m_board1->Location = System::Drawing::Point(2, m_ribbonHeight + menuStrip1->Height);
            m_board2->Location = System::Drawing::Point(2 + boardWidth*BOARD_SQUARE_SIZE + m_sepWidth/2, m_ribbonHeight + menuStrip1->Height);

            m_seq = gcnew SequentialEvolution(m_board1->m_matrix, m_board1);
            m_par = gcnew ParallelEvolution(m_board2->m_matrix, m_board2);

            m_seqThread = gcnew Thread(gcnew ThreadStart(m_seq, &SequentialEvolution::Run));
            m_parThread = gcnew Thread(gcnew ThreadStart(m_par, &ParallelEvolution::Run));        

            Thread::CurrentThread->Priority = ThreadPriority::AboveNormal;

            m_suspend = true;
        }
    protected:
        ~Form1()
        {
            if (components)
            {
                delete components;
            }
        }
    private: System::Windows::Forms::MenuStrip^  menuStrip1;
    private: System::Windows::Forms::ToolStripMenuItem^  fileToolStripMenuItem;
    private: System::Windows::Forms::ToolStripMenuItem^  exitToolStripMenuItem;
    private: System::Windows::Forms::ToolStripMenuItem^  gameToolStripMenuItem;
    private: System::Windows::Forms::ToolStripMenuItem^  seedToolStripMenuItem;
    private: System::Windows::Forms::ToolStripMenuItem^  runToolStripMenuItem;
    private: System::Windows::Forms::ToolStripMenuItem^  pauseToolStripMenuItem;
    private: Board^ m_board1;
    private: Board^ m_board2;
    private: System::Windows::Forms::Label^  Sequential;
    private: System::Windows::Forms::Label^  label1;
    private: static const int m_sepWidth = 5;
    private: static const int m_ribbonHeight = 26;
    private: SequentialEvolution^ m_seq;
    private: ParallelEvolution^ m_par;
    private: Thread^ m_seqThread;
    private: Thread^ m_parThread;
    private: System::Windows::Forms::Label^  seqGen;
    private: System::Windows::Forms::Label^  parGen;
    private: bool m_suspend;

    private:
        System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
        void InitializeComponent(void)
        {
            this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
            this->fileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
            this->exitToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
            this->gameToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
            this->seedToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
            this->runToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
            this->pauseToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
            this->Sequential = (gcnew System::Windows::Forms::Label());
            this->label1 = (gcnew System::Windows::Forms::Label());
            this->seqGen = (gcnew System::Windows::Forms::Label());
            this->parGen = (gcnew System::Windows::Forms::Label());
            this->menuStrip1->SuspendLayout();
            this->SuspendLayout();
            // 
            // menuStrip1
            // 
            this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) 
                {this->fileToolStripMenuItem, this->gameToolStripMenuItem});
            this->menuStrip1->Location = System::Drawing::Point(0, 0);
            this->menuStrip1->Name = L"menuStrip1";
            this->menuStrip1->Padding = System::Windows::Forms::Padding(8, 2, 0, 2);
            this->menuStrip1->Size = System::Drawing::Size(1600, 26);
            this->menuStrip1->TabIndex = 0;
            this->menuStrip1->Text = L"menuStrip1";
            this->menuStrip1->ItemClicked += gcnew System::Windows::Forms::ToolStripItemClickedEventHandler(this, &Form1::menuStrip1_ItemClicked);
            // 
            // fileToolStripMenuItem
            // 
            this->fileToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) {this->exitToolStripMenuItem});
            this->fileToolStripMenuItem->Name = L"fileToolStripMenuItem";
            this->fileToolStripMenuItem->Size = System::Drawing::Size(40, 22);
            this->fileToolStripMenuItem->Text = L"File";
            // 
            // exitToolStripMenuItem
            // 
            this->exitToolStripMenuItem->Name = L"exitToolStripMenuItem";
            this->exitToolStripMenuItem->Size = System::Drawing::Size(99, 22);
            this->exitToolStripMenuItem->Text = L"Exit";
            this->exitToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::OnExit);
            // 
            // gameToolStripMenuItem
            // 
            this->gameToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {this->seedToolStripMenuItem, 
                this->runToolStripMenuItem, this->pauseToolStripMenuItem});
            this->gameToolStripMenuItem->Name = L"gameToolStripMenuItem";
            this->gameToolStripMenuItem->Size = System::Drawing::Size(59, 22);
            this->gameToolStripMenuItem->Text = L"Game";
            // 
            // seedToolStripMenuItem
            // 
            this->seedToolStripMenuItem->Name = L"seedToolStripMenuItem";
            this->seedToolStripMenuItem->Size = System::Drawing::Size(115, 22);
            this->seedToolStripMenuItem->Text = L"Seed";
            this->seedToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::OnSeed);
            // 
            // runToolStripMenuItem
            // 
            this->runToolStripMenuItem->Enabled = false;
            this->runToolStripMenuItem->Name = L"runToolStripMenuItem";
            this->runToolStripMenuItem->Size = System::Drawing::Size(115, 22);
            this->runToolStripMenuItem->Text = L"Run";
            this->runToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::OnRun);
            // 
            // pauseToolStripMenuItem
            // 
            this->pauseToolStripMenuItem->Enabled = false;
            this->pauseToolStripMenuItem->Name = L"pauseToolStripMenuItem";
            this->pauseToolStripMenuItem->Size = System::Drawing::Size(115, 22);
            this->pauseToolStripMenuItem->Text = L"Pause";
            this->pauseToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::OnPauseResume);
            // 
            // Sequential
            // 
            this->Sequential->AutoSize = true;
            this->Sequential->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
                static_cast<System::Byte>(0)));
            this->Sequential->Location = System::Drawing::Point(12, 32);
            this->Sequential->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
            this->Sequential->Name = L"Sequential";
            this->Sequential->Size = System::Drawing::Size(239, 18);
            this->Sequential->TabIndex = 1;
            this->Sequential->Text = L"Sequential Algorithm      generation:";
            // 
            // label1
            // 
            this->label1->AutoSize = true;
            this->label1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
                static_cast<System::Byte>(0)));
            this->label1->Location = System::Drawing::Point(813, 32);
            this->label1->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
            this->label1->Name = L"label1";
            this->label1->Size = System::Drawing::Size(219, 18);
            this->label1->TabIndex = 2;
            this->label1->Text = L"Parallel Algorithm     generation: ";
            // 
            // seqGen
            // 
            this->seqGen->AutoSize = true;
            this->seqGen->Location = System::Drawing::Point(289, 35);
            this->seqGen->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
            this->seqGen->Name = L"seqGen";
            this->seqGen->Size = System::Drawing::Size(16, 17);
            this->seqGen->TabIndex = 3;
            this->seqGen->Text = L"0";
            // 
            // parGen
            // 
            this->parGen->AutoSize = true;
            this->parGen->Location = System::Drawing::Point(1068, 35);
            this->parGen->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
            this->parGen->Name = L"parGen";
            this->parGen->Size = System::Drawing::Size(16, 17);
            this->parGen->TabIndex = 4;
            this->parGen->Text = L"0";
            // 
            // Form1
            // 
            this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
            this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
            this->ClientSize = System::Drawing::Size(1600, 738);
            this->Controls->Add(this->parGen);
            this->Controls->Add(this->seqGen);
            this->Controls->Add(this->label1);
            this->Controls->Add(this->Sequential);
            this->Controls->Add(this->menuStrip1);
            this->MainMenuStrip = this->menuStrip1;
            this->Margin = System::Windows::Forms::Padding(4);
            this->MaximizeBox = false;
            this->Name = L"Form1";
            this->Text = L"Game of Life";
            this->menuStrip1->ResumeLayout(false);
            this->menuStrip1->PerformLayout();
            this->ResumeLayout(false);
            this->PerformLayout();

        }
#pragma endregion    
    protected: 
        void CloseApp ()
        {
            m_seq->Quit();
            m_par->Quit();
            //! Perform a very ungracious exit, should coordinate the threads
            System::Environment::Exit(0);            
        }
    
    protected: 
        virtual void OnPaint(PaintEventArgs^ e) override
        {
        }

        virtual void OnFormClosing(FormClosingEventArgs^ e) override
        { 
            CloseApp();
        }
    
        void OnExit(System::Object^ sender, System::EventArgs^ e)
        {                
            CloseApp();
        }

        void OnSeed(System::Object^ sender, System::EventArgs^ e)
        {
            this->seedToolStripMenuItem->Enabled = false;
            this->runToolStripMenuItem->Enabled = true;            
            time_t now = time(NULL);
            this->m_board1->seed((int)now);
            this->m_board2->seed(this->m_board1);
            this->Invalidate();
        }

        void OnRun(System::Object^ sender, System::EventArgs^ e)
        {    
            this->runToolStripMenuItem->Enabled = false;        
            this->pauseToolStripMenuItem->Enabled = true;
            m_seqThread->Start();
            m_parThread->Start();    
        }

        void OnPauseResume(System::Object^ sender, System::EventArgs^ e)
        {    
            if (m_suspend)
            {
                m_seq->SetPause(true);
                m_par->SetPause(true);
                this->pauseToolStripMenuItem->Text = L"Resume";
            }
            else
            {
                m_seq->SetPause(false);
                m_par->SetPause(false);            
                this->pauseToolStripMenuItem->Text = L"Pause";
            }
            m_suspend = !m_suspend;
        }

    private: 
        System::Void menuStrip1_ItemClicked(System::Object^  sender, System::Windows::Forms::ToolStripItemClickedEventArgs^  e) 
        {}
};
#endif
#endif
