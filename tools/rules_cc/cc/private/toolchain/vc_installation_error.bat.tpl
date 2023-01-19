:: Copyright 2017 The Bazel Authors. All rights reserved.
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::    http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.

@echo OFF

echo. 1>&2
echo The target you are compiling requires Visual C++ build tools. 1>&2
echo Bazel couldn't find a valid Visual C++ build tools installation on your machine. 1>&2
%{vc_error_message}
echo Please check your installation following https://docs.bazel.build/versions/master/windows.html#using 1>&2
echo. 1>&2

exit /b 1
