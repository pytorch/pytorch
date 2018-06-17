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

// Internal Intel tool

#ifndef __STATISTICS_H__
#define __STATISTICS_H__

#define _CRT_SECURE_NO_DEPRECATE 1

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <time.h>

using namespace std;
typedef double value_t;

/*
   Statistical collector class.

   Resulting table output:
        +---------------------------------------------------------------------------+
        | [Date] <Title>...                                                         |
        +----------+----v----+--v---+----------------+------------+-..-+------------+
        | TestName | Threads | Mode | Rounds results | Stat_type1 | .. | Stat_typeN |
        +----------+---------+------+-+-+-+-..-+-+-+-+------------+-..-+------------+
        |          |         |      | | | | .. | | | |            |    |            |
        ..        ...       ...     ..................            ......           ..
        |          |         |      | | | | .. | | | |            |    |            |
        +----------+---------+------+-+-+-+-..-+-+-+-+------------+-..-+------------+

   Iterating table output:
        +---------------------------------------------------------------------------+
        | [Date] <TestName>, Threads: <N>, Mode: <M>; for <Title>...                |
        +----------+----v----+--v---+----------------+------------+-..-+------------+

*/

class StatisticsCollector
{
public:
    typedef map<string, string> Analysis_t;
    typedef vector<value_t> Results_t;

protected:
    StatisticsCollector(const StatisticsCollector &);

    struct StatisticResults
    {
        string              Name;
        string              Mode;
        int                 Threads;
        Results_t           Results;
        Analysis_t          Analysis;
    };

    // internal members
    //bool OpenFile;
    StatisticResults *CurrentKey;
    string Title;
    const char /**Name,*/ *ResultsFmt;
    string Name;
    //! Data
    typedef map<string, StatisticResults*> Statistics_t;
    Statistics_t Statistics;
    typedef vector<string> RoundTitles_t;
    RoundTitles_t RoundTitles;
    //TODO: merge those into one structure
    typedef map<string, string> Formulas_t;
    Formulas_t   Formulas;
    typedef set<string> AnalysisTitles_t;
    AnalysisTitles_t AnalysisTitles;
    typedef vector<pair<string, string> > RunInfo_t;
    RunInfo_t RunInfo;

public:
    struct TestCase {
        StatisticResults *access;
        TestCase() : access(0) {}
        TestCase(StatisticResults *link) : access(link) {}
        const char *getName() const { return access->Name.c_str(); }
        const char *getMode() const { return access->Mode.c_str(); }
        int getThreads()       const { return access->Threads; }
        const Results_t &getResults() const { return access->Results; }
        const Analysis_t &getAnalysis() const { return access->Analysis; }
    };

    enum Sorting {
        ByThreads, ByAlg
    };

    //! Data and output types
    enum DataOutput {
        // Verbosity level enumeration
        Statistic = 1,     //< Analytical data - computed after all iterations and rounds passed
        Result    = 2,     //< Testing data    - collected after all iterations passed
        Iteration = 3,     //< Verbose data    - collected at each iteration (for each size - in case of containers)
        // ExtraVerbose is not applicabe yet :) be happy, but flexibility is always welcome

        // Next constants are bit-fields
        Stdout   = 1<<8,    //< Output to the console
        TextFile = 1<<9,    //< Output to plain text file "name.txt" (delimiter is TAB by default)
        ExcelXML = 1<<10,   //< Output to Excel-readable XML-file "name.xml"
        HTMLFile = 1<<11,   //< Output to HTML file "name.html"
        PivotMode= 1<<15    //< Puts all the rounds into one columt to better fit for pivot table in Excel
    };

    //! Constructor. Specify tests set name which used as name of output files
    StatisticsCollector(const char *name, Sorting mode = ByThreads, const char *fmt = "%g")
        :  CurrentKey(NULL), ResultsFmt(fmt), Name(name), SortMode(mode) {}

    ~StatisticsCollector();

    //! Set tests set title, supporting printf-like arguments
    void SetTitle(const char *fmt, ...);

    //! Specify next test key
    TestCase SetTestCase(const char *name, const char *mode, int threads);
    //! Specify next test key
    void SetTestCase(const TestCase &t) { SetTestCase(t.getName(), t.getMode(), t.getThreads()); }
    //! Reserve specified number of rounds. Use for efficiency. Used mostly internally
    void ReserveRounds(size_t index);
    //! Add result of the measure
    void AddRoundResult(const TestCase &, value_t v);
    //! Add result of the current measure
    void AddRoundResult(value_t v) { if(CurrentKey) AddRoundResult(TestCase(CurrentKey), v); }
    //! Add title of round
    void SetRoundTitle(size_t index, const char *fmt, ...);
    //! Add numbered title of round
    void SetRoundTitle(size_t index, int num) { SetRoundTitle(index, "%d", num); }
    //! Get number of rounds
    size_t GetRoundsCount() const { return RoundTitles.size(); }
    // Set statistic value for the test
    void AddStatisticValue(const TestCase &, const char *type, const char *fmt, ...);
    // Set statistic value for the current test
    void AddStatisticValue(const char *type, const char *fmt, ...);
    //! Add Excel-processing formulas. @arg formula can contain more than one instances of
    //! ROUNDS template which transforms into the range of cells with result values
    //TODO://! #1 .. #n templates represent data cells from the first to the last
    //TODO: merge with Analisis
    void SetStatisticFormula(const char *name, const char *formula);
    //! Add information about run or compile parameters
    void SetRunInfo(const char *title, const char *fmt, ...);
    void SetRunInfo(const char *title, int num) { SetRunInfo(title, "%d", num); }

    //! Data output
    void Print(int dataOutput, const char *ModeName = "Mode");

private:
    Sorting SortMode;
};

//! using: Func(const char *fmt, ...) { vargf2buff(buff, 128, fmt);...
#define vargf2buff(name, size, fmt) \
    char name[size]; memset(name, 0, size); \
    va_list args; va_start(args, fmt); \
    vsnprintf(name, size-1, fmt, args); \
    va_end(args);


inline std::string Format(const char *fmt, ...) {
    vargf2buff(buf, 1024, fmt); // from statistics.h
    return std::string(buf);
}

#ifdef STATISTICS_INLINE
#include "statistics.cpp"
#endif
#endif //__STATISTICS_H__
