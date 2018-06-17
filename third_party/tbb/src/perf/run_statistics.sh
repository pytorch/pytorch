#!/bin/bash
#
# Copyright (c) 2005-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
#

export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
#setting output format .csv, 'pivot' - is pivot table mode, ++ means append
export STAT_FORMAT=pivot-csv++
#check existing files because of apend mode
ls *.csv
rm -i *.csv
#setting a delimiter in txt or csv file
#export STAT_DELIMITER=,
export STAT_RUNINFO1=Host=`hostname -s`
#append a suffix after the filename
#export STAT_SUFFIX=$STAT_RUNINFO1
for ((i=1;i<=${repeat:=100};++i)); do echo $i of $repeat: && STAT_RUNINFO2=Run=$i $* || break; done
