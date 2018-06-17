@echo off
REM
REM Copyright (c) 2005-2018 Intel Corporation
REM
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM     http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.
REM
REM
REM
REM
REM

:: Getting parameters
if ("%1") == ("") goto error0
if ("%2") == ("") goto error0
if ("%3") == ("") goto error0
set arch=%1
if ("%2") == ("debug") set postfix=_debug
set output_dir=%3

:: Optional 4th parameter to set install root
if ("%4") NEQ ("") set TBBROOT=%4
:: Actually we can set install root by ourselves
if ("%TBBROOT%") == ("") set TBBROOT=%~d0%~p0..\..\

:: Getting vs folders in case vc_mt binaries are not provided
:: ordered from oldest to newest, so we end with newest available version
if ("%VS110COMNTOOLS%") NEQ ("") set vc_dir=vc11
if ("%VS120COMNTOOLS%") NEQ ("") set vc_dir=vc12
if ("%VS140COMNTOOLS%") NEQ ("") set vc_dir=vc14
:: To use Microsoft* Visual Studio* 2017 IDE, make sure the variable VS150COMNTOOLS is set in your IDE instance.
:: If it is not, try running Microsoft Visual Studio 2017 from Microsoft* Developer Command Prompt* for VS 2017.
:: For details, see https://developercommunity.visualstudio.com/content/problem/730/vs154-env-var-vs150comntools-missing-from-build-sy.html
if ("%VS150COMNTOOLS%") NEQ ("") set vc_dir=vc14

:: Are we standalone/oss or inside compiler?
if exist "%TBBROOT%\bin\%arch%\%vc_dir%\tbb%postfix%.dll" set interim_path=bin\%arch%
if exist "%TBBROOT%\..\redist\%arch%\tbb\%vc_dir%\tbb%postfix%.dll" set interim_path=..\redist\%arch%\tbb
if ("%interim_path%") == ("") goto error1

:: Do we provide vc_mt binaries?
if exist "%TBBROOT%\%interim_path%\vc_mt\tbb%postfix%.dll" set vc_dir=vc_mt
if ("%vc_dir%") == ("") goto error2

:: We know everything we wanted and there are no errors
:: Copying binaries

copy "%TBBROOT%\%interim_path%\%vc_dir%\tbb%postfix%.dll" "%output_dir%"
copy "%TBBROOT%\%interim_path%\%vc_dir%\tbb%postfix%.pdb" "%output_dir%"
copy "%TBBROOT%\%interim_path%\%vc_dir%\tbbmalloc%postfix%.dll" "%output_dir%"
copy "%TBBROOT%\%interim_path%\%vc_dir%\tbbmalloc%postfix%.pdb" "%output_dir%"
if exist "%TBBROOT%\%interim_path%\%vc_dir%\tbb_preview%postfix%.dll" copy "%TBBROOT%\%interim_path%\%vc_dir%\tbb_preview%postfix%.dll" "%output_dir%"
if exist "%TBBROOT%\%interim_path%\%vc_dir%\tbb_preview%postfix%.pdb" copy "%TBBROOT%\%interim_path%\%vc_dir%\tbb_preview%postfix%.pdb" "%output_dir%"

goto end
:error0
echo number of parameters not correct
exit /B 1
:error1
echo Could not determine path to TBB libraries
exit /B 1
:error2
echo Could not determine Visual Studio version
exit /B 1

:end
exit /B 0

