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

__constant int redChannelOffset = 0;
__constant int greenChannelOffset = 1;
__constant int blueChannelOffset = 2;
__constant int channelsPerPixel = 4;
__constant int channelIncreaseValue = 10;

__kernel void mergeImages( __global uchar* bufferLeft, __global uchar* bufferRight, uint width) {
    const int indexWidth = get_global_id(0);
    const int indexHeight = get_global_id(1);

    const int pixelIndex = channelsPerPixel * width * indexHeight + channelsPerPixel * indexWidth;
    const int pixelGreenChannelIndex = pixelIndex + greenChannelOffset;
    const int pixelBlueChannelIndex = pixelIndex + blueChannelOffset;

    bufferLeft[pixelGreenChannelIndex] = (bufferRight[pixelGreenChannelIndex] + bufferLeft[pixelGreenChannelIndex]) / 2;
    bufferLeft[pixelBlueChannelIndex] = bufferRight[pixelBlueChannelIndex];
}

__kernel void applyLeftImageEffect( __global uchar* bufferLeft, uint width) {
    const int indexWidth = get_global_id(0);
    const int indexHeight = get_global_id(1);

    const int pixelRedChannelIndex = channelsPerPixel * width * indexHeight + channelsPerPixel * indexWidth + redChannelOffset;

    bufferLeft[pixelRedChannelIndex] += channelIncreaseValue;
}

__kernel void applyRightImageEffect( __global uchar* bufferRight, uint width) {
    const int indexWidth = get_global_id(0);
    const int indexHeight = get_global_id(1);

    const int pixelBlueChannelIndex = channelsPerPixel * width * indexHeight + channelsPerPixel * indexWidth + blueChannelOffset;

    bufferRight[pixelBlueChannelIndex] += channelIncreaseValue;

}
