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

#include <iostream>
#include <string>

#include "lodepng.h"

namespace utils {

    struct image_buffer {
        unsigned int width, height;
        std::shared_ptr< std::vector<unsigned char> > buffer; // smart pointer to the vector of raw pixels in RGBA format, 4 bytes per pixel
    };

    image_buffer readPNGImage(const std::string& imageName) {
        image_buffer image;
        image.buffer = std::make_shared< std::vector<unsigned char> >();

        unsigned int error = lodepng::decode(*image.buffer, image.width, image.height, imageName.c_str());

        if (error) {
            std::string exceptionMessage = "decoder error: " + std::string(lodepng_error_text(error));
            throw std::runtime_error(exceptionMessage);
        }
        return image;
    }

    void readPNGImage(unsigned char* imageBuffer, unsigned int& width, unsigned int& height, const char* imageName) {
        unsigned int error = lodepng_decode32_file(&imageBuffer, &width, &height, imageName);

        if (error) {
            std::string exceptionMessage = "decoder error: " + std::string(lodepng_error_text(error));
            throw std::runtime_error(exceptionMessage);
        }
    }

    void writePNGImage(const image_buffer& image, const std::string& outputFile) {
        unsigned int error = lodepng::encode(outputFile, *image.buffer, image.width, image.height);
        if (error) {
            std::string exceptionMessage = "encoder error: " + std::string(lodepng_error_text(error));
            throw std::runtime_error(exceptionMessage);
        }
    }

    void writePNGImage(unsigned char* imageBuffer, unsigned int& width, unsigned int& height, const std::string& outputFile) {
        unsigned int error = lodepng::encode(outputFile, imageBuffer, width, height);
        if (error) {
            std::string exceptionMessage = "encoder error: " + std::string(lodepng_error_text(error));
            throw std::runtime_error(exceptionMessage);
        }
    }

    image_buffer generatePNGImage() {
        image_buffer image;
        image.width = 1024;
        image.height = 1024;

        image.buffer = std::make_shared< std::vector<unsigned char> >(image.width * image.height * 4);
        std::vector<unsigned char>& buffer = *image.buffer;

        const int widthOffset = 4 * image.width;
        for (unsigned y = 0; y < image.height; y++) {
            for (unsigned x = 0; x < image.width; x++) {
                const int pixelOffset = widthOffset * y + 4 * x;
                buffer[pixelOffset] = 200 * !(x & y);
                buffer[pixelOffset + 1] = x ^ y;
                buffer[pixelOffset + 2] = x | y;
                buffer[pixelOffset + 3] = 255;
            }
        }
        return image;
    }

    bool isFileExist(const std::string& fileName) {
        std::ifstream file(fileName);
        return file.good();
    }

    bool isBothImagesExists(const std::string& firstFile, const std::string& secondFile) {
        return isFileExist(firstFile) && isFileExist(secondFile);
    }

    image_buffer getOrGenerateImage(const std::string& fileName) {
        return fileName.empty() ? generatePNGImage() : readPNGImage(fileName);
    }
}
