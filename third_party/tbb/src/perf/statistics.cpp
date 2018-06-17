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

#include "statistics.h"
#include "statistics_xml.h"

#define COUNT_PARAMETERS 3

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

void GetTime(char* buff,int size_buff) 
{
    tm *newtime;
    time_t timer;
    time(&timer);
    newtime=localtime(&timer);
    strftime(buff,size_buff,"%H:%M:%S",newtime); 
}

void GetDate(char* buff,int size_buff) 
{
    tm *newtime;
    time_t timer;
    time(&timer);  
    newtime=localtime(&timer);
    strftime(buff,size_buff,"%Y-%m-%d",newtime); 
}


StatisticsCollector::TestCase StatisticsCollector::SetTestCase(const char *name, const char *mode, int threads)
{
    string KeyName(name);
    switch (SortMode)
    {
    case ByThreads: KeyName += Format("_%02d_%s", threads, mode); break;
    default:
    case ByAlg: KeyName += Format("_%s_%02d", mode, threads); break;
    }
    CurrentKey = Statistics[KeyName];
    if(!CurrentKey) {
        CurrentKey = new StatisticResults;
        CurrentKey->Mode = mode;
        CurrentKey->Name = name;
        CurrentKey->Threads = threads;
        CurrentKey->Results.reserve(RoundTitles.size());
        Statistics[KeyName] = CurrentKey;
    }
    return TestCase(CurrentKey);
}

StatisticsCollector::~StatisticsCollector()
{
    for(Statistics_t::iterator i = Statistics.begin(); i != Statistics.end(); i++)
        delete i->second;
}

void StatisticsCollector::ReserveRounds(size_t index)
{
    size_t i = RoundTitles.size();
    if (i > index) return;
    char buf[16];
    RoundTitles.resize(index+1);
    for(; i <= index; i++) {
        snprintf( buf, 15, "%u", unsigned(i+1) );
        RoundTitles[i] = buf;
    }
    for(Statistics_t::iterator i = Statistics.begin(); i != Statistics.end(); i++) {
        if(!i->second) printf("!!!'%s' = NULL\n", i->first.c_str());
        else i->second->Results.reserve(index+1);
    }
}

void StatisticsCollector::AddRoundResult(const TestCase &key, value_t v)
{
    ReserveRounds(key.access->Results.size());
    key.access->Results.push_back(v);
}

void StatisticsCollector::SetRoundTitle(size_t index, const char *fmt, ...)
{
    vargf2buff(buff, 128, fmt);
    ReserveRounds(index);
    RoundTitles[index] = buff;
}

void StatisticsCollector::AddStatisticValue(const TestCase &key, const char *type, const char *fmt, ...)
{
    vargf2buff(buff, 128, fmt);
    AnalysisTitles.insert(type);
    key.access->Analysis[type] = buff;
}

void StatisticsCollector::AddStatisticValue(const char *type, const char *fmt, ...)
{
    vargf2buff(buff, 128, fmt);
    AnalysisTitles.insert(type);
    CurrentKey->Analysis[type] = buff;
}

void StatisticsCollector::SetRunInfo(const char *title, const char *fmt, ...)
{
    vargf2buff(buff, 256, fmt);
    RunInfo.push_back(make_pair(title, buff));
}

void StatisticsCollector::SetStatisticFormula(const char *name, const char *formula)
{
    Formulas[name] = formula;
}

void StatisticsCollector::SetTitle(const char *fmt, ...)
{
    vargf2buff(buff, 256, fmt);
    Title = buff;
}

string ExcelFormula(const string &fmt, size_t place, size_t rounds, bool is_horizontal)
{
    char buff[16];
    if(is_horizontal)
        snprintf(buff, 15, "RC[%u]:RC[%u]", unsigned(place), unsigned(place+rounds-1));
    else
        snprintf(buff, 15, "R[%u]C:R[%u]C", unsigned(place+1), unsigned(place+rounds));
    string result(fmt); size_t pos = 0;
    while ( (pos = result.find("ROUNDS", pos, 6)) != string::npos )
        result.replace(pos, 6, buff);
    return result;
}

void StatisticsCollector::Print(int dataOutput, const char *ModeName)
{
    FILE *OutputFile;
    const char *file_suffix = getenv("STAT_SUFFIX");
    if( !file_suffix ) file_suffix = "";
    const char *file_format = getenv("STAT_FORMAT");
    if( file_format ) {
        dataOutput = 0;
        if( strstr(file_format, "con")||strstr(file_format, "std") ) dataOutput |= StatisticsCollector::Stdout;
        if( strstr(file_format, "txt")||strstr(file_format, "csv") ) dataOutput |= StatisticsCollector::TextFile;
        if( strstr(file_format, "excel")||strstr(file_format, "xml") ) dataOutput |= StatisticsCollector::ExcelXML;
        if( strstr(file_format, "htm") ) dataOutput |= StatisticsCollector::HTMLFile;
        if( strstr(file_format, "pivot") ) dataOutput |= StatisticsCollector::PivotMode;
    }
    for(int i = 1; i < 10; i++) {
        string env = Format("STAT_RUNINFO%d", i);
        const char *info = getenv(env.c_str());
        if( info ) {
            string title(info);
            size_t pos = title.find('=');
            if( pos != string::npos ) {
                env = title.substr(pos+1);
                title.resize(pos);
            } else env = title;
            RunInfo.push_back(make_pair(title, env));
        }
    }

    if (dataOutput & StatisticsCollector::Stdout)
    {
        printf("\n-=# %s #=-\n", Title.c_str());
        if(SortMode == ByThreads)
            printf("    Name    |  #  | %s ", ModeName);
        else
            printf("    Name    | %s |  #  ", ModeName);
        for (AnalysisTitles_t::iterator i = AnalysisTitles.begin(); i != AnalysisTitles.end(); i++)
            printf("|%s", i->c_str()+1);

        for (Statistics_t::iterator i = Statistics.begin(); i != Statistics.end(); i++)
        {
            if(SortMode == ByThreads)
                printf("\n%12s|% 5d|%6s", i->second->Name.c_str(), i->second->Threads, i->second->Mode.c_str());
            else
                printf("\n%12s|%6s|% 5d", i->second->Name.c_str(), i->second->Mode.c_str(), i->second->Threads);
            Analysis_t &analisis = i->second->Analysis;
            AnalysisTitles_t::iterator t = AnalysisTitles.begin();
            for (Analysis_t::iterator a = analisis.begin(); a != analisis.end(); t++)
            {
                char fmt[8]; snprintf(fmt, 7, "|%% %us", unsigned(max(size_t(3), t->size())));
                if(*t != a->first)
                    printf(fmt, "");
                else {
                    printf(fmt, a->second.c_str()); a++;
                }
            }
        }
        printf("\n");
    }
    if (dataOutput & StatisticsCollector::TextFile)
    {
        bool append = false;
        const char *file_ext = ".txt";
        if( file_format && strstr(file_format, "++") ) append = true;
        if( file_format && strstr(file_format, "csv") ) file_ext = ".csv";
        if ((OutputFile = fopen((Name+file_suffix+file_ext).c_str(), append?"at":"wt")) == NULL) {
            printf("Can't open .txt file\n");
        } else {
            const char *delim = getenv("STAT_DELIMITER");
            if( !delim || !delim[0] ) {
                if( file_format && strstr(file_format, "csv") ) delim = ",";
                else delim = "\t";
            }
            if( !append || !ftell(OutputFile) ) { // header needed
                append = false;
                if(SortMode == ByThreads) fprintf(OutputFile, "Name%s#%s%s", delim, delim, ModeName);
                else fprintf(OutputFile, "Name%s%s%s#", delim, ModeName, delim);
                for( size_t k = 0; k < RunInfo.size(); k++ )
                    fprintf(OutputFile, "%s%s", delim, RunInfo[k].first.c_str());
            }
            if(dataOutput & StatisticsCollector::PivotMode) {
                if( !append) fprintf(OutputFile, "%sColumn%sValue", delim, delim);
                for (Statistics_t::iterator i = Statistics.begin(); i != Statistics.end(); i++)
                {
                    string RowHead;
                    if(SortMode == ByThreads)
                        RowHead = Format("\n%s%s%d%s%s%s", i->second->Name.c_str(), delim, i->second->Threads, delim, i->second->Mode.c_str(), delim);
                    else
                        RowHead = Format("\n%s%s%s%s%d%s", i->second->Name.c_str(), delim, i->second->Mode.c_str(), delim, i->second->Threads, delim);
                    for( size_t k = 0; k < RunInfo.size(); k++ )
                        RowHead.append(RunInfo[k].second + delim);
                    Analysis_t &analisis = i->second->Analysis;
                    for (Analysis_t::iterator a = analisis.begin(); a != analisis.end(); ++a)
                        fprintf(OutputFile, "%s%s%s%s", RowHead.c_str(), a->first.c_str(), delim, a->second.c_str());
                    Results_t &r = i->second->Results;
                    for (size_t k = 0; k < r.size(); k++) {
                        fprintf(OutputFile, "%s%s%s", RowHead.c_str(), RoundTitles[k].c_str(), delim);
                        fprintf(OutputFile, ResultsFmt, r[k]);
                    }
                }
            } else {
                if( !append ) {
                    for( size_t k = 0; k < RunInfo.size(); k++ )
                        fprintf(OutputFile, "%s%s", delim, RunInfo[k].first.c_str());
                    for (AnalysisTitles_t::iterator i = AnalysisTitles.begin(); i != AnalysisTitles.end(); i++)
                        fprintf(OutputFile, "%s%s", delim, i->c_str()+1);
                    for (size_t i = 0; i < RoundTitles.size(); i++)
                        fprintf(OutputFile, "%s%s", delim, RoundTitles[i].c_str());
                }
                for (Statistics_t::iterator i = Statistics.begin(); i != Statistics.end(); i++)
                {
                    if(SortMode == ByThreads)
                        fprintf(OutputFile, "\n%s%s%d%s%s", i->second->Name.c_str(), delim, i->second->Threads, delim, i->second->Mode.c_str());
                    else
                        fprintf(OutputFile, "\n%s%s%s%s%d", i->second->Name.c_str(), delim, i->second->Mode.c_str(), delim, i->second->Threads);
                    for( size_t k = 0; k < RunInfo.size(); k++ )
                        fprintf(OutputFile, "%s%s", delim, RunInfo[k].second.c_str());
                    Analysis_t &analisis = i->second->Analysis;
                    AnalysisTitles_t::iterator t = AnalysisTitles.begin();
                    for (Analysis_t::iterator a = analisis.begin(); a != analisis.end(); ++t) {
                        fprintf(OutputFile, "%s", delim);
                        if(*t == a->first) {
                            fprintf(OutputFile, "%s", a->second.c_str()); ++a;
                        }
                    }
                    //data
                    Results_t &r = i->second->Results;
                    for (size_t k = 0; k < r.size(); k++)
                    {
                        fprintf(OutputFile, "%s", delim);
                        fprintf(OutputFile, ResultsFmt, r[k]);
                    }
                }
            }
            fprintf(OutputFile, "\n");
            fclose(OutputFile);
        }
    }
    if (dataOutput & StatisticsCollector::HTMLFile)
    {
        if ((OutputFile = fopen((Name+file_suffix+".html").c_str(), "w+t")) == NULL) {
            printf("Can't open .html file\n");
        } else {
            char TimerBuff[100], DateBuff[100];
            GetTime(TimerBuff,sizeof(TimerBuff));
            GetDate(DateBuff,sizeof(DateBuff));
            fprintf(OutputFile, "<html><head>\n<title>%s</title>\n</head><body>\n", Title.c_str());
            //-----------------------
            fprintf(OutputFile, "<table id=\"h\" style=\"position:absolute;top:20\" border=1 cellspacing=0 cellpadding=2>\n");
            fprintf(OutputFile, "<tr><td><a name=hr href=#vr onclick=\"v.style.visibility='visible';"
                                "h.style.visibility='hidden';\">Flip[H]</a></td>"
                                "<td>%s</td><td>%s</td><td colspan=%u>%s",
                DateBuff, TimerBuff, unsigned(AnalysisTitles.size() + RoundTitles.size()), Title.c_str());
            for( size_t k = 0; k < RunInfo.size(); k++ )
                fprintf(OutputFile, "; %s: %s", RunInfo[k].first.c_str(), RunInfo[k].second.c_str());
            fprintf(OutputFile, "</td></tr>\n<tr bgcolor=#CCFFFF><td>Name</td><td>Threads</td><td>%s</td>", ModeName);
            for (AnalysisTitles_t::iterator i = AnalysisTitles.begin(); i != AnalysisTitles.end(); i++)
                fprintf(OutputFile, "<td>%s</td>", i->c_str()+1);
            for (size_t i = 0; i < RoundTitles.size(); i++)
                fprintf(OutputFile, "<td>%s</td>", RoundTitles[i].c_str());
            for (Statistics_t::iterator i = Statistics.begin(); i != Statistics.end(); i++)
            {
                fprintf(OutputFile, "</tr>\n<tr><td bgcolor=#CCFFCC>%s</td><td bgcolor=#CCFFCC>%d</td><td bgcolor=#CCFFCC>%4s</td>",
                    i->second->Name.c_str(), i->second->Threads, i->second->Mode.c_str());
                //statistics
                AnalysisTitles_t::iterator t = AnalysisTitles.begin();
                for (Analysis_t::iterator j = i->second->Analysis.begin(); j != i->second->Analysis.end(); t++)
                {
                    fprintf(OutputFile, "<td bgcolor=#FFFF99>%s</td>", (*t != j->first)?" ":(i->second->Analysis[j->first]).c_str());
                    if(*t == j->first) j++;
                }
                //data
                Results_t &r = i->second->Results;
                for (size_t k = 0; k < r.size(); k++)
                {
                    fprintf(OutputFile, "<td>");
                    fprintf(OutputFile, ResultsFmt, r[k]);
                    fprintf(OutputFile, "</td>");
                }
            }
            fprintf(OutputFile, "</tr>\n</table>\n");
            //////////////////////////////////////////////////////
            fprintf(OutputFile, "<table id=\"v\" style=\"visibility:hidden;position:absolute;top:20\" border=1 cellspacing=0 cellpadding=2>\n");
            fprintf(OutputFile, "<tr><td><a name=vr href=#hr onclick=\"h.style.visibility='visible';"
                                "v.style.visibility='hidden';\">Flip[V]</a></td>\n"
                                "<td>%s</td><td>%s</td><td colspan=%u>%s</td>", 
                DateBuff, TimerBuff, unsigned(max(Statistics.size()-2,size_t(1))), Title.c_str());

            fprintf(OutputFile, "</tr>\n<tr bgcolor=#CCFFCC><td bgcolor=#CCFFFF>Name</td>");
            for (Statistics_t::iterator i = Statistics.begin(); i != Statistics.end(); i++)
                fprintf(OutputFile, "<td>%s</td>", i->second->Name.c_str());
            fprintf(OutputFile, "</tr>\n<tr bgcolor=#CCFFCC><td bgcolor=#CCFFFF>Threads</td>");
            for (Statistics_t::iterator n = Statistics.begin(); n != Statistics.end(); n++)
                fprintf(OutputFile, "<td>%d</td>", n->second->Threads);
            fprintf(OutputFile, "</tr>\n<tr bgcolor=#CCFFCC><td bgcolor=#CCFFFF>%s</td>", ModeName);
            for (Statistics_t::iterator m = Statistics.begin(); m != Statistics.end(); m++)
                fprintf(OutputFile, "<td>%s</td>", m->second->Mode.c_str());

            for (AnalysisTitles_t::iterator t = AnalysisTitles.begin(); t != AnalysisTitles.end(); t++)
            {
                fprintf(OutputFile, "</tr>\n<tr bgcolor=#FFFF99><td bgcolor=#CCFFFF>%s</td>", t->c_str()+1);
                for (Statistics_t::iterator i = Statistics.begin(); i != Statistics.end(); i++)
                    fprintf(OutputFile, "<td>%s</td>", i->second->Analysis.count(*t)?i->second->Analysis[*t].c_str():" ");
            }

            for (size_t r = 0; r < RoundTitles.size(); r++)
            {
                fprintf(OutputFile, "</tr>\n<tr><td bgcolor=#CCFFFF>%s</td>", RoundTitles[r].c_str());
                for (Statistics_t::iterator i = Statistics.begin(); i != Statistics.end(); i++)
                {
                    Results_t &result = i->second->Results;
                    fprintf(OutputFile, "<td>");
                    if(result.size() > r)
                        fprintf(OutputFile, ResultsFmt, result[r]);
                    fprintf(OutputFile, "</td>");
                }
            }
            fprintf(OutputFile, "</tr>\n</table>\n</body></html>\n");
            fclose(OutputFile);
        }
    }
    if (dataOutput & StatisticsCollector::ExcelXML)
    {
        if ((OutputFile = fopen((Name+file_suffix+".xml").c_str(), "w+t")) == NULL) {
            printf("Can't open .xml file\n");
        } else {
            // TODO:PivotMode
            char UserName[100];
            char TimerBuff[100], DateBuff[100];
#if _WIN32 || _WIN64
            strcpy(UserName,getenv("USERNAME"));
#else
            strcpy(UserName,getenv("USER"));
#endif
            //--------------------------------
            GetTime(TimerBuff,sizeof(TimerBuff));
            GetDate(DateBuff,sizeof(DateBuff));
            //--------------------------
            fprintf(OutputFile, XMLHead, UserName, TimerBuff);
            fprintf(OutputFile, XMLStyles);
            fprintf(OutputFile, XMLBeginSheet, "Horizontal");
            fprintf(OutputFile, XMLNames,1,1,1,int(AnalysisTitles.size()+Formulas.size()+COUNT_PARAMETERS));
            fprintf(OutputFile, XMLBeginTable, int(RoundTitles.size()+Formulas.size()+AnalysisTitles.size()+COUNT_PARAMETERS+1/*title*/), int(Statistics.size()+1));
            fprintf(OutputFile, XMLBRow);
            fprintf(OutputFile, XMLCellTopName);
            fprintf(OutputFile, XMLCellTopThread);
            fprintf(OutputFile, XMLCellTopMode, ModeName);
            for (AnalysisTitles_t::iterator j = AnalysisTitles.begin(); j != AnalysisTitles.end(); j++)
                fprintf(OutputFile, XMLAnalysisTitle, j->c_str()+1);
            for (Formulas_t::iterator j = Formulas.begin(); j != Formulas.end(); j++)
                fprintf(OutputFile, XMLAnalysisTitle, j->first.c_str()+1);
            for (RoundTitles_t::iterator j = RoundTitles.begin(); j != RoundTitles.end(); j++)
                fprintf(OutputFile, XMLAnalysisTitle, j->c_str());
            string Info = Title;
            for( size_t k = 0; k < RunInfo.size(); k++ )
                Info.append("; " + RunInfo[k].first + "=" + RunInfo[k].second);
            fprintf(OutputFile, XMLCellEmptyWhite, Info.c_str());
            fprintf(OutputFile, XMLERow);
            //------------------------
            for (Statistics_t::iterator i = Statistics.begin(); i != Statistics.end(); i++)
            {
                fprintf(OutputFile, XMLBRow);
                fprintf(OutputFile, XMLCellName,  i->second->Name.c_str());
                fprintf(OutputFile, XMLCellThread,i->second->Threads);
                fprintf(OutputFile, XMLCellMode,  i->second->Mode.c_str());
                //statistics
                AnalysisTitles_t::iterator at = AnalysisTitles.begin();
                for (Analysis_t::iterator j = i->second->Analysis.begin(); j != i->second->Analysis.end(); at++)
                {
                    fprintf(OutputFile, XMLCellAnalysis, (*at != j->first)?"":(i->second->Analysis[j->first]).c_str());
                    if(*at == j->first) j++;
                }
                //formulas
                size_t place = 0;
                Results_t &v = i->second->Results;
                for (Formulas_t::iterator f = Formulas.begin(); f != Formulas.end(); f++, place++)
                    fprintf(OutputFile, XMLCellFormula, ExcelFormula(f->second, Formulas.size()-place, v.size(), true).c_str());
                //data
                for (size_t k = 0; k < v.size(); k++)
                {
                    fprintf(OutputFile, XMLCellData, v[k]);
                }
                if(v.size() < RoundTitles.size())
                    fprintf(OutputFile, XMLMergeRow, int(RoundTitles.size() - v.size()));
                fprintf(OutputFile, XMLERow);
            }
            //------------------------
            fprintf(OutputFile, XMLEndTable);
            fprintf(OutputFile, XMLWorkSheetProperties,1,1,3,3,int(RoundTitles.size()+AnalysisTitles.size()+Formulas.size()+COUNT_PARAMETERS));
            fprintf(OutputFile, XMLAutoFilter,1,1,1,int(AnalysisTitles.size()+Formulas.size()+COUNT_PARAMETERS));
            fprintf(OutputFile, XMLEndWorkSheet);
            //----------------------------------------
            fprintf(OutputFile, XMLEndWorkbook);
            fclose(OutputFile);
        }
    }
}
