/*
    Copyright (c) 2017-2018 Intel Corporation

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

#include "ipc_utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>

namespace tbb {
namespace internal {
namespace rml {

#define MAX_STR_LEN 255
#define STARTTIME_ITEM_ID 21

static char* get_stat_item(char* line, int item_id) {
    int id = 0, i = 0;

    while( id!=item_id ) {
        while( line[i]!='(' && line[i]!=' ' && line[i]!='\0' ) {
            ++i;
        }
        if( line[i]==' ' ) {
            ++id;
            ++i;
        } else if( line[i]=='(' ) {
            while( line[i]!=')' && line[i]!='\0' ) {
               ++i;
            }
            if( line[i]==')' ) {
                ++i;
            } else {
                return NULL;
            }
        } else {
            return NULL;
        }
    }

    return line + i;
}

unsigned long long get_start_time(int pid) {
    const char* stat_file_path_template = "/proc/%d/stat";
    char stat_file_path[MAX_STR_LEN + 1];
    sprintf( stat_file_path, stat_file_path_template, pid );

    FILE* stat_file = fopen( stat_file_path, "rt" );
    if( stat_file==NULL ) {
        return 0;
    }

    char stat_line[MAX_STR_LEN + 1];
    char* line = fgets( stat_line, MAX_STR_LEN, stat_file );
    if( line==NULL ) {
        return 0;
    }

    char* starttime_str = get_stat_item( stat_line, STARTTIME_ITEM_ID );
    if( starttime_str==NULL ) {
        return 0;
    }

    unsigned long long starttime = strtoull( starttime_str, NULL, 10 );
    if( starttime==ULLONG_MAX ) {
        return 0;
    }

    return starttime;
}

char* get_shared_name(const char* prefix, int pid, unsigned long long time) {
    const char* name_template = "%s_%d_%llu";
    const int digits_in_int = 10;
    const int digits_in_long = 20;

    int len = strlen( name_template ) + strlen( prefix ) + digits_in_int + digits_in_long + 1;
    char* name = new char[len];
    sprintf( name, name_template, prefix, pid, time );

    return name;
}

char* get_shared_name(const char* prefix) {
    int pid = getpgrp();
    unsigned long long time = get_start_time( pid );
    return get_shared_name( prefix, pid, time );
}

int get_num_threads(const char* env_var) {
    if( env_var==NULL ) {
        return 0;
    }

    char* value = getenv( env_var );
    if( value==NULL ) {
        return 0;
    }

    int num_threads = (int)strtol( value, NULL, 10 );
    return num_threads;
}

bool get_enable_flag(const char* env_var) {
    if( env_var==NULL ) {
        return false;
    }

    char* value = getenv( env_var );
    if( value==NULL ) {
        return false;
    }

    if( strcmp( value, "0" ) == 0 ||
        strcmp( value, "false" ) == 0 ||
        strcmp( value, "False" ) == 0 ||
        strcmp( value, "FALSE" ) == 0 ) {
        return false;
    }

    return true;
}

}}} //tbb::internal::rml
