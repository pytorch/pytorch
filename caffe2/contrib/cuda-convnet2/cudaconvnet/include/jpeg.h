/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef JPEG_MAIN_H
#define JPEG_MAIN_H

#include <cstdio>
#include <cstdlib>
#include <Python.h>
#include <vector>
#include <string>
#include <iostream>
#include <jpeglib.h>
//#include <arrayobject.h>
#include "../../util/include/thread.h"
#include "../../util/include/matrix.h"

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

#define NUM_JPEG_DECODER_THREADS        4


class DecoderThread : public Thread {
 protected:
    PyObject* _pyList;
    Matrix* _target;
    int64 _start_img, _end_img;
    int64 _img_size, _inner_size, _inner_pixels;
    bool _test, _multiview;

    unsigned char* _decodeTarget;
    int64 _decodeTargetSize;
    unsigned int _rseed;

    void* run();
    void decodeJpeg(int idx, int& width, int& height);
    double randUniform();
    double randUniform(double min, double max);
    void crop(int64 i, int64 width, int64 height, bool flip);
    virtual void crop(int64 i, int64 src_width, int64 src_height, bool flip, int64 crop_start_x, int64 crop_start_y);
 public:
    DecoderThread(PyObject* pyList, Matrix& target, int start_img, int end_img, int img_size, int inner_size, bool test, bool multiview);
    virtual ~DecoderThread();
};

#endif // JPEG_MAIN_H
