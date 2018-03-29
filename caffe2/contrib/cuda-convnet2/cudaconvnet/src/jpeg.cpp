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

#include "../include/jpeg.h"

using namespace std;

/* ========================
 * DecoderThread
 * ========================
 */
DecoderThread::DecoderThread(PyObject* pyList, Matrix& target, int start_img, int end_img, int img_size, int inner_size, bool test, bool multiview)
: Thread(true), _pyList(pyList), _target(&target), _start_img(start_img), _end_img(end_img),
  _img_size(img_size), _inner_size(inner_size), _test(test), _multiview(multiview),
  _decodeTarget(0), _decodeTargetSize(0) {

    _inner_pixels = _inner_size * _inner_size;
    _rseed = time(0);
}

DecoderThread::~DecoderThread(){
    free(_decodeTarget);
}

void* DecoderThread::run() {
    int numSrcCases = PyList_GET_SIZE(_pyList);
    assert(_target->getNumCols() == _inner_pixels * 3);
    assert(_target->getNumRows() == PyList_GET_SIZE(_pyList) * (_multiview ? 10 : 1));

    int width, height;

    for (int64 i = _start_img; i < _end_img; ++i) {
        decodeJpeg(i, width, height);
        assert((width == _img_size && height >= _img_size)
               || (height == _img_size && width >= _img_size));
        if (_multiview) {
            for (int flip = 0; flip < 2; ++flip) {
                crop(numSrcCases * (flip * 5 + 0) + i, width, height, flip, 0, 0); // top-left
                crop(numSrcCases * (flip * 5 + 1) + i, width, height, flip, width - _inner_size, 0); // top-right
                crop(numSrcCases * (flip * 5 + 2) + i, width, height, flip, (width - _inner_size) / 2, (height - _inner_size) / 2); // center
                crop(numSrcCases * (flip * 5 + 3) + i, width, height, flip, 0, height - _inner_size); // bottom-left
                crop(numSrcCases * (flip * 5 + 4) + i, width, height, flip, width - _inner_size, height - _inner_size); // bottom-right
            }
        } else {
            crop(i, width, height, !_test && (rand_r(&_rseed) % 2));
        }

    }
    return NULL;
}

void DecoderThread::decodeJpeg(int idx, int& width, int& height) {
    PyObject* pySrc = PyList_GET_ITEM(_pyList, idx);
    unsigned char* src = (unsigned char*)PyString_AsString(pySrc);
    size_t src_len = PyString_GET_SIZE(pySrc);
    
    struct jpeg_decompress_struct cinf;
    struct jpeg_error_mgr jerr;
    cinf.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinf);
    jpeg_mem_src(&cinf, src, src_len);
    assert(jpeg_read_header(&cinf, TRUE));
    cinf.out_color_space = JCS_RGB;
    assert(jpeg_start_decompress(&cinf));
    assert(cinf.num_components == 3 || cinf.num_components == 1);
    width = cinf.image_width;
    height = cinf.image_height;

    if (_decodeTargetSize < width * height * 3) {
        free(_decodeTarget);
        _decodeTargetSize = width * height * 3 * 3;
        _decodeTarget = (unsigned char*)malloc(_decodeTargetSize);
    }
    
    while (cinf.output_scanline < cinf.output_height) {
        JSAMPROW tmp = &_decodeTarget[width * cinf.out_color_components * cinf.output_scanline];
        assert(jpeg_read_scanlines(&cinf, &tmp, 1) > 0);
    }
    assert(jpeg_finish_decompress(&cinf));
    jpeg_destroy_decompress(&cinf);
}

/*
 * Uniform in [0,1)
 */
inline double DecoderThread::randUniform() {
    return double(rand_r(&_rseed)) / (int64(RAND_MAX) + 1);
}

/*
 * Uniform in [min, max)
 */
inline double DecoderThread::randUniform(double min, double max) {
    return (max - min) * randUniform() + min;
}

void DecoderThread::crop(int64 i, int64 src_width, int64 src_height, bool flip) {
    crop(i, src_width, src_height, flip, -1, -1);
}

void DecoderThread::crop(int64 i, int64 src_width, int64 src_height, bool flip, int64 crop_start_x, int64 crop_start_y) {
    const int64 border_size_y = src_height - _inner_size;
    const int64 border_size_x = src_width - _inner_size;
    if (crop_start_x < 0) {
        crop_start_x = _test ? (border_size_x / 2) : (rand_r(&_rseed) % (border_size_x + 1));
    }
    if (crop_start_y < 0) {
        crop_start_y = _test ? (border_size_y / 2) : (rand_r(&_rseed) % (border_size_y + 1));
    }
    const int64 src_pixels = src_width * src_height;
    for (int64 c = 0; c < 3; ++c) {
        for (int64 y = crop_start_y; y < crop_start_y + _inner_size; ++y) {
            for (int64 x = crop_start_x; x < crop_start_x + _inner_size; ++x) {
                assert((y >= 0 && y < src_height && x >= 0 && x < src_width));
                _target->getCell(i, c * _inner_pixels + (y - crop_start_y) * _inner_size
                                    + (flip ? (_inner_size - 1 - x + crop_start_x)
                                        : (x - crop_start_x)))
                        = _decodeTarget[3 * (y * src_width + x) + c];
            }
        }
    }
}