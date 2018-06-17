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

const char XMLBRow[]=
"   <Row>\n";

const char XMLERow[]=
"   </Row>\n";

const char XMLHead[]=
"<?xml version=\"1.0\"?>\n"
"<?mso-application progid=\"Excel.Sheet\"?>\n\
<Workbook xmlns=\"urn:schemas-microsoft-com:office:spreadsheet\"\n\
 xmlns:o=\"urn:schemas-microsoft-com:office:office\"\n\
 xmlns:x=\"urn:schemas-microsoft-com:office:excel\"\n\
 xmlns:ss=\"urn:schemas-microsoft-com:office:spreadsheet\"\n\
 xmlns:html=\"http://www.w3.org/TR/REC-html40\">\n\
 <DocumentProperties xmlns=\"urn:schemas-microsoft-com:office:office\">\n\
  <Author>%s</Author>\n\
  <Created>%s</Created>\n\
  <Company>Intel Corporation</Company>\n\
 </DocumentProperties>\n\
 <ExcelWorkbook xmlns=\"urn:schemas-microsoft-com:office:excel\">\n\
  <RefModeR1C1/>\n\
 </ExcelWorkbook>\n";
 
 const char XMLStyles[]=
 " <Styles>\n\
  <Style ss:ID=\"Default\" ss:Name=\"Normal\">\n\
   <Alignment ss:Vertical=\"Bottom\" ss:Horizontal=\"Left\" ss:WrapText=\"0\"/>\n\
  </Style>\n\
  <Style ss:ID=\"s26\">\n\
   <Alignment ss:Vertical=\"Top\"  ss:Horizontal=\"Left\" ss:WrapText=\"0\"/>\n\
   <Borders>\n\
    <Border ss:Position=\"Bottom\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
    <Border ss:Position=\"Left\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
    <Border ss:Position=\"Right\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
    <Border ss:Position=\"Top\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
   </Borders>\n\
   <Interior ss:Color=\"#FFFF99\" ss:Pattern=\"Solid\"/>\n\
  </Style>\n\
  <Style ss:ID=\"s25\">\n\
   <Alignment ss:Vertical=\"Top\"  ss:Horizontal=\"Left\" ss:WrapText=\"0\"/>\n\
   <Borders>\n\
    <Border ss:Position=\"Bottom\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
    <Border ss:Position=\"Left\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
    <Border ss:Position=\"Right\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
    <Border ss:Position=\"Top\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
   </Borders>\n\
   <Interior ss:Color=\"#CCFFFF\" ss:Pattern=\"Solid\"/>\n\
  </Style>\n\
  <Style ss:ID=\"s24\">\n\
   <Alignment ss:Vertical=\"Top\"  ss:Horizontal=\"Left\" ss:WrapText=\"0\"/>\n\
   <Borders>\n\
    <Border ss:Position=\"Bottom\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
    <Border ss:Position=\"Left\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
    <Border ss:Position=\"Right\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
    <Border ss:Position=\"Top\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
   </Borders>\n\
   <Interior ss:Color=\"#CCFFCC\" ss:Pattern=\"Solid\"/>\n\
  </Style>\n\
  <Style ss:ID=\"s23\">\n\
   <Alignment ss:Vertical=\"Top\"  ss:Horizontal=\"Left\" ss:WrapText=\"0\"/>\n\
   <Borders>\n\
    <Border ss:Position=\"Bottom\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
    <Border ss:Position=\"Left\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
    <Border ss:Position=\"Right\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
    <Border ss:Position=\"Top\" ss:LineStyle=\"Continuous\" ss:Weight=\"1\"/>\n\
   </Borders>\n\
  </Style>\n\
 </Styles>\n";

const char XMLBeginSheet[]=
" <Worksheet ss:Name=\"%s\">\n";

const char XMLNames[]=
"  <Names>\n\
   <NamedRange ss:Name=\"_FilterDatabase\" ss:RefersTo=\"R%dC%d:R%dC%d\" ss:Hidden=\"1\"/>\n\
  </Names>\n";

const char XMLBeginTable[]=
"  <Table ss:ExpandedColumnCount=\"%d\" ss:ExpandedRowCount=\"%d\" x:FullColumns=\"1\"\n\
   x:FullRows=\"1\">\n";
   
const char XMLColumsHorizontalTable[]=
"   <Column ss:Index=\"1\" ss:Width=\"108.75\"/>\n\
   <Column ss:Index=\"%d\" ss:Width=\"77.25\" ss:Span=\"%d\"/>\n";
 
const char XMLColumsVerticalTable[]= 
"   <Column ss:Index=\"1\" ss:Width=\"77.25\" ss:Span=\"%d\"/>\n";

const char XMLNameAndTime[]=
"    <Cell><Data ss:Type=\"String\">%s</Data></Cell>\n\
    <Cell><Data ss:Type=\"String\">%s</Data></Cell>\n\
    <Cell><Data ss:Type=\"String\">%s</Data></Cell>\n";

const char XMLTableParamAndTitle[]=
"    <Cell><Data ss:Type=\"Number\">%d</Data></Cell>\n\
    <Cell><Data ss:Type=\"Number\">%d</Data></Cell>\n\
    <Cell><Data ss:Type=\"Number\">%d</Data></Cell>\n\
    <Cell><Data ss:Type=\"String\">%s</Data></Cell>\n";

//--------------
const char XMLCellTopName[]=
"   <Cell ss:StyleID=\"s25\"><Data ss:Type=\"String\">Name</Data></Cell>\n";
const char XMLCellTopThread[]=
"   <Cell ss:StyleID=\"s25\"><Data ss:Type=\"String\">Threads</Data></Cell>\n";
const char XMLCellTopMode[]=
"   <Cell ss:StyleID=\"s25\"><Data ss:Type=\"String\">%s</Data></Cell>\n";
//---------------------
const char XMLAnalysisTitle[]=
"   <Cell ss:StyleID=\"s25\"><Data ss:Type=\"String\">%s</Data></Cell>\n";

const char XMLCellName[]=
"    <Cell ss:StyleID=\"s24\"><Data ss:Type=\"String\">%s</Data></Cell>\n";

const char XMLCellThread[]=
"    <Cell ss:StyleID=\"s24\"><Data ss:Type=\"Number\">%d</Data></Cell>\n";

const char XMLCellMode[]=
"    <Cell ss:StyleID=\"s24\"><Data ss:Type=\"String\">%s</Data></Cell>\n";

const char XMLCellAnalysis[]=
"    <Cell ss:StyleID=\"s26\"><Data ss:Type=\"String\">%s</Data></Cell>\n";

const char XMLCellFormula[]=
"    <Cell ss:StyleID=\"s26\" ss:Formula=\"%s\"><Data ss:Type=\"Number\"></Data></Cell>\n";

const char XMLCellData[]=
"    <Cell ss:StyleID=\"s23\"><Data ss:Type=\"Number\">%g</Data></Cell>\n";

const char XMLMergeRow[]=
"   <Cell ss:StyleID=\"s23\" ss:MergeAcross=\"%d\" ><Data ss:Type=\"String\"></Data></Cell>\n";

const char XMLCellEmptyWhite[]=
"    <Cell><Data ss:Type=\"String\">%s</Data></Cell>\n";

const char XMLCellEmptyTitle[]=
"    <Cell ss:StyleID=\"s25\"><Data ss:Type=\"String\"></Data></Cell>\n";

const char XMLEndTable[]=
"  </Table>\n";

const char XMLAutoFilter[]=
"  <AutoFilter x:Range=\"R%dC%d:R%dC%d\" xmlns=\"urn:schemas-microsoft-com:office:excel\">\n\
  </AutoFilter>\n";

const char XMLEndWorkSheet[]=
 " </Worksheet>\n";

const char XMLWorkSheetProperties[]=
"  <WorksheetOptions xmlns=\"urn:schemas-microsoft-com:office:excel\">\n\
   <Unsynced/>\n\
   <Selected/>\n\
   <FreezePanes/>\n\
   <FrozenNoSplit/>\n\
   <SplitHorizontal>%d</SplitHorizontal>\n\
   <TopRowBottomPane>%d</TopRowBottomPane>\n\
   <SplitVertical>%d</SplitVertical>\n\
   <LeftColumnRightPane>%d</LeftColumnRightPane>\n\
   <ActivePane>0</ActivePane>\n\
   <Panes>\n\
    <Pane>\n\
     <Number>3</Number>\n\
    </Pane>\n\
    <Pane>\n\
     <Number>1</Number>\n\
    </Pane>\n\
    <Pane>\n\
     <Number>2</Number>\n\
    </Pane>\n\
    <Pane>\n\
     <Number>0</Number>\n\
     <ActiveRow>0</ActiveRow>\n\
     <ActiveCol>%d</ActiveCol>\n\
    </Pane>\n\
   </Panes>\n\
   <ProtectObjects>False</ProtectObjects>\n\
   <ProtectScenarios>False</ProtectScenarios>\n\
  </WorksheetOptions>\n";

const char XMLEndWorkbook[]=
 "</Workbook>\n";
