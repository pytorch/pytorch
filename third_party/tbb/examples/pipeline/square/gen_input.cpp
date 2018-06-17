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

#include <stdlib.h>
#include <stdio.h>
#include <stdexcept>

#if _WIN32
#include <io.h>
#ifndef F_OK
#define F_OK 0
#endif
#define access _access
#else
#include <unistd.h>
#endif

const long INPUT_SIZE = 1000000;

//! Generates sample input for square.cpp
void gen_input( const char *fname ) {
    long num = INPUT_SIZE;
    FILE *fptr = fopen(fname, "w");
    if(!fptr) {
        throw std::runtime_error("Could not open file for generating input");
    }

    int a=0;
    int b=1;
    for( long j=0; j<num; ++j ) {
        fprintf(fptr, "%u\n",a);
        b+=a;
        a=(b-a)%10000;
        if (a<0) a=-a;
    }

    if(fptr) {
        fclose(fptr);
    }
}

void generate_if_needed( const char *fname ) {
    if ( access(fname, F_OK) != 0 )
        gen_input(fname);
}
