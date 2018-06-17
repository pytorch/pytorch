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
if "%DXSDK_DIR%"=="" goto error_no_DXSDK
goto end

:error_no_DXSDK
echo DirectX SDK Check : error : This example requires the DirectX SDK.  Either (re)-install the DirectX SDK, or set the DXSDK_DIR environment variable to indicate where it is installed.

:end

