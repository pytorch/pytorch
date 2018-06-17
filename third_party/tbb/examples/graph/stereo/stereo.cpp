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

#define TBB_PREVIEW_FLOW_GRAPH_NODES 1
#define TBB_PREVIEW_FLOW_GRAPH_FEATURES 1

#include "tbb/tbb_config.h"
#include "../../common/utility/utility.h"

#if __TBB_PREVIEW_OPENCL_NODE && __TBB_CPP11_LAMBDAS_PRESENT

#if _MSC_VER
// suppress warning C4503: decorated name length exceeded, name was truncated 
#pragma warning(disable : 4503)
#endif

#include <iostream>

#include "tbb/flow_graph.h"
#include "tbb/flow_graph_opencl_node.h"
#include "tbb/tick_count.h"

#include "utils.h"

static const int redChannelOffset = 0;
static const int greenChannelOffset = 1;
static const int blueChannelOffset = 2;
static const int channelsPerPixel = 4;
static const int channelIncreaseValue = 10;

void applyLeftImageEffect(utils::image_buffer& image) {
    const int heighBase = channelsPerPixel * image.width;
    std::vector<unsigned char>& buffer = *image.buffer;

    // Increase the Red channel of left image by 10
    for (unsigned int y = 0; y < image.height; y++) {
        const int heightOffset = heighBase * y;
        for (unsigned int x = 0; x < image.width; x++) {
            int pixelOffset = heightOffset + channelsPerPixel * x + redChannelOffset;
            buffer[pixelOffset] += channelIncreaseValue;
        }
    }
}

void applyRightImageEffect(utils::image_buffer& image) {
    const int heighBase = channelsPerPixel * image.width;
    std::vector<unsigned char>& buffer = *image.buffer;

    // Increase the Blue channel of left image by 10
    for (unsigned int y = 0; y < image.height; y++) {
        const int heightOffset = heighBase * y;
        for (unsigned int x = 0; x < image.width; x++) {
            const int pixelOffset = heightOffset + channelsPerPixel * x + blueChannelOffset;
            buffer[pixelOffset] += channelIncreaseValue;
        }
    }
}

// This function merges to image buffers into the first buffer (leftImageBuffer as a destination)
void mergeImageBuffers(utils::image_buffer& leftImage, const utils::image_buffer& rightImage) {
    const int heighBase = channelsPerPixel * leftImage.width;
    std::vector<unsigned char>& leftImageBuffer = *leftImage.buffer;
    std::vector<unsigned char>& rightImageBuffer = *rightImage.buffer;

    // Apply stereoscopic merge using algorithm: R: left image, G: left and right images (middle value), B: right image
    for (unsigned int y = 0; y < leftImage.height; y++) {
        const int heightOffset = heighBase * y;
        for (unsigned int x = 0; x < leftImage.width; x++) {
            const int pixelOffset = heightOffset + channelsPerPixel * x;
            const int greenChannelIndex = pixelOffset + greenChannelOffset;
            const int blueChannelIndex = pixelOffset + blueChannelOffset;
            const int middleGreenChannel = (leftImageBuffer[greenChannelIndex] + rightImageBuffer[greenChannelIndex]);
            leftImageBuffer[greenChannelIndex] = middleGreenChannel / 2;
            leftImageBuffer[blueChannelIndex] = rightImageBuffer[blueChannelIndex];
        }
    }
}

void fillOpenclBuffer(tbb::flow::opencl_buffer<cl_uchar>& openclBuffer, const std::vector<unsigned char>& sourceBuffer) {
    std::copy(sourceBuffer.begin(), sourceBuffer.end(), openclBuffer.begin());
}

class gpu_device_selector {
public:
    template <typename DeviceFilter>
    tbb::flow::opencl_device operator()(tbb::flow::opencl_factory<DeviceFilter>& f) {
        // Set your GPU device if available to execute kernel on
        const tbb::flow::opencl_device_list &devices = f.devices();
        tbb::flow::opencl_device_list::const_iterator it = std::find_if(
            devices.cbegin(), devices.cend(),
            [](const tbb::flow::opencl_device &d) {
            cl_device_type type;
            d.info(CL_DEVICE_TYPE, type);
            return CL_DEVICE_TYPE_GPU == type;
        });

        if (it == devices.cend()) {
            std::cout << "Info: could not find any GPU devices. Choosing the first available device (default behaviour)." << std::endl;
            return *(f.devices().begin());
        } else {
            // Return GPU device from factory
            return *it;
        }
    }
};

// Image processing function that is executed on CPU only
void hostFunction(const std::string& firstFile, const std::string& secondFile, const std::string& outputFile) {
    using namespace tbb::flow;
    typedef tuple< utils::image_buffer, utils::image_buffer > MergeImagesTuple;

    graph g;

    function_node< std::string, utils::image_buffer > fileReaderOne(g, serial, [](const std::string& fileToRead) -> utils::image_buffer {
        return utils::getOrGenerateImage(fileToRead);
    });

    function_node< std::string, utils::image_buffer > fileReaderTwo = fileReaderOne;

    function_node< utils::image_buffer, utils::image_buffer > leftImageEffect(g, unlimited, [](utils::image_buffer image) -> utils::image_buffer {
        applyLeftImageEffect(image);
        return image;
    });

    function_node< utils::image_buffer, utils::image_buffer > rightImageEffect(g, unlimited, [](utils::image_buffer image) -> utils::image_buffer {
        applyRightImageEffect(image);
        return image;
    });

    join_node< tuple< utils::image_buffer, utils::image_buffer > > joinNode(g);

    function_node< MergeImagesTuple, utils::image_buffer > mergeImages(g, unlimited, [](const MergeImagesTuple& bufferTuple) -> utils::image_buffer {
        // Two input images from tupple are merged into the first image,
        utils::image_buffer leftImageBuffer = std::get<0>(bufferTuple);
        utils::image_buffer rightImageBuffer = std::get<1>(bufferTuple);

        mergeImageBuffers(leftImageBuffer, rightImageBuffer);

        return leftImageBuffer;
    });

    function_node< utils::image_buffer > outputWriter(g, unlimited, [&outputFile](const utils::image_buffer& image) {
        utils::writePNGImage(image, outputFile);
    });

    // Read left image
    make_edge(fileReaderOne, leftImageEffect);

    // Read right image
    make_edge(fileReaderTwo, rightImageEffect);

    // Process left image
    make_edge(leftImageEffect, tbb::flow::input_port<0>(joinNode));

    // Process right image
    make_edge(rightImageEffect, tbb::flow::input_port<1>(joinNode));

    // Merge images
    make_edge(joinNode, mergeImages);
    make_edge(mergeImages, outputWriter);

    // Start graph image processing
    fileReaderOne.try_put(firstFile);
    fileReaderTwo.try_put(secondFile);

    g.wait_for_all();
}


// Image processing function using OpenCL
/** Reading and writing image to file is executed on CPU, while all buffers manipulation are executed on GPU */
void openclFunctionGPU(const std::string& firstFile, const std::string& secondFile, const std::string& outputFile) {
    using namespace tbb::flow;
    typedef opencl_buffer<cl_uchar> OpenclImageBuffer;
    typedef std::array<unsigned int, 2> NDRange;
    typedef tuple< OpenclImageBuffer, cl_uint, NDRange > OpenclImageTuple;
    typedef tuple< OpenclImageBuffer, OpenclImageBuffer, cl_uint, NDRange > OpenclImagesMergeTuple;
    typedef tuple< OpenclImageBuffer, NDRange > WriteImageBufferTuple;

    graph g;

    gpu_device_selector gpu_selector;

    function_node< std::string, OpenclImageTuple > fileReaderOne(g, serial, [&g](const std::string& fileToRead) -> OpenclImageTuple {
        utils::image_buffer src = utils::getOrGenerateImage(fileToRead);

        // Create and initialize opencl_buffer in order to pass it to kernel
        OpenclImageBuffer oclImage(src.buffer->size());
        fillOpenclBuffer(oclImage, *src.buffer);

        NDRange rangeList = { src.width, src.height };
        return std::make_tuple(oclImage, src.width, rangeList);
    });

    function_node< std::string, OpenclImageTuple > fileReaderTwo = fileReaderOne;

    split_node< OpenclImageTuple > splitArgumentsLeftNode(g);

    // Kernel should be in the current folder
    opencl_program<> program("imageEffects.cl");

    opencl_node< OpenclImageTuple > leftImageEffect(g, program.get_kernel("applyLeftImageEffect"), gpu_selector);

    split_node< OpenclImageTuple > splitArgumentsRightNode(g);

    opencl_node< OpenclImageTuple > rightImageEffect(g, program.get_kernel("applyRightImageEffect"), gpu_selector);

    opencl_node< OpenclImagesMergeTuple > mergeImages(g, program.get_kernel("mergeImages"), gpu_selector);

    join_node< WriteImageBufferTuple > joinTupleNode(g);

    function_node< WriteImageBufferTuple > outputWriter(g, unlimited, [&outputFile](const WriteImageBufferTuple& image) {
        // The result image have to be copied in order to be changed,
        // the second parameter - image size, can be taken by const reference
        OpenclImageBuffer imageBuffer = std::get<0>(image);
        const NDRange& imageSize = std::get<1>(image);
        unsigned int width = imageSize[0];
        unsigned int height = imageSize[1]; 

        utils::writePNGImage(imageBuffer.data(), width, height, outputFile);
    });

    // Process left image
    make_edge(fileReaderOne, splitArgumentsLeftNode);
    make_edge(output_port<0>(splitArgumentsLeftNode), input_port<0>(leftImageEffect));
    make_edge(output_port<1>(splitArgumentsLeftNode), input_port<1>(leftImageEffect));

    // Pass OpenCL NDRange via input port because it depends on input data
    make_edge(output_port<2>(splitArgumentsLeftNode), input_port<2>(leftImageEffect));

    // Process right image
    make_edge(fileReaderTwo, splitArgumentsRightNode);
    make_edge(output_port<0>(splitArgumentsRightNode), input_port<0>(rightImageEffect));
    make_edge(output_port<1>(splitArgumentsRightNode), input_port<1>(rightImageEffect));

    // Pass OpenCL NDRange via input port because it depends on input data
    make_edge(output_port<2>(splitArgumentsRightNode), input_port<2>(rightImageEffect));

    // Merge images
    make_edge(output_port<0>(leftImageEffect), input_port<0>(mergeImages));
    make_edge(output_port<0>(rightImageEffect), input_port<1>(mergeImages));
    make_edge(output_port<1>(leftImageEffect), input_port<2>(mergeImages));

    // Set OpenCL NDRange here (because the values may vary, depending on input data)
    make_edge(output_port<2>(leftImageEffect), input_port<3>(mergeImages));

    // Write image to PNG
    make_edge(output_port<0>(mergeImages), input_port<0>(joinTupleNode));
    make_edge(output_port<3>(mergeImages), input_port<1>(joinTupleNode));
    make_edge(joinTupleNode, outputWriter);

    // Define where to get ndrange and kernel arguments
    leftImageEffect.set_args(port_ref<0, 1>());
    leftImageEffect.set_range(port_ref<2>());

    rightImageEffect.set_args(port_ref<0, 1>());
    rightImageEffect.set_range(port_ref<2>());

    mergeImages.set_args(port_ref<0, 2>());
    mergeImages.set_range(port_ref<3>());

    // Start graph image processing pipeline
    fileReaderOne.try_put(firstFile);
    fileReaderTwo.try_put(secondFile);

    g.wait_for_all();
}


// Second image processing function using OpenCL
/** Reading and writing image to file is executed on CPU, while some buffers manipulation are executed on GPU
    and others runs on CPU device. This case should have the best performance among others. */
void openclFunctionGPUPlusCPU(const std::string& firstFile, const std::string& secondFile, const std::string& outputFile) {
    using namespace tbb::flow;
    typedef opencl_buffer<cl_uchar> OpenclImageBuffer;
    typedef std::array<unsigned int, 2> NDRange;
    typedef tuple< OpenclImageBuffer, cl_uint, NDRange > OpenclImageTuple;
    typedef tuple< OpenclImageBuffer, OpenclImageBuffer, cl_uint, NDRange > OpenclImagesMergeTuple;
    typedef tuple< OpenclImageBuffer, NDRange > WriteImageBufferTuple;

    graph g;

    gpu_device_selector gpu_selector;

    function_node< std::string, OpenclImageTuple > fileReaderOne(g, serial, [&g](const std::string& fileToRead) -> OpenclImageTuple {
        utils::image_buffer src = utils::getOrGenerateImage(fileToRead);

        // Create and initialize opencl_buffer in order to pass it to mergeImages kernel
        OpenclImageBuffer oclImage(src.buffer->size());
        fillOpenclBuffer(oclImage, *src.buffer);

        NDRange rangeList = { src.width, src.height };
        return std::make_tuple(oclImage, src.width, rangeList);
    });

    function_node< std::string, utils::image_buffer > fileReaderTwo(g, serial, [](const std::string& fileToRead) -> utils::image_buffer {
        return utils::readPNGImage(fileToRead);
    });

    split_node< OpenclImageTuple > splitArgumentsLeftNode(g);

    // Kernel should be in the current folder
    opencl_program<> program("imageEffects.cl");

    opencl_node< OpenclImageTuple > leftImageEffect(g, program.get_kernel("applyLeftImageEffect"), gpu_selector);

    function_node< utils::image_buffer, OpenclImageBuffer > rightImageEffect(g, unlimited, [&g](utils::image_buffer image) -> OpenclImageBuffer {
        applyRightImageEffect(image);

        // Create and initialize opencl_buffer in order to pass it to kernel
        OpenclImageBuffer oclImage(image.buffer->size());
        fillOpenclBuffer(oclImage, *image.buffer);

        return oclImage;
    });

    opencl_node< OpenclImagesMergeTuple > mergeImages(g, program.get_kernel("mergeImages"), gpu_selector);

    join_node< WriteImageBufferTuple > joinTupleNode(g);

    function_node< WriteImageBufferTuple > outputWriter(g, unlimited, [&outputFile](const WriteImageBufferTuple& image) {
        // The result image have to be copied in order to be changed,
        // the second parameter - image size, can be taken by const reference
        OpenclImageBuffer imageBuffer = std::get<0>(image);
        const NDRange& imageSize = std::get<1>(image);
        unsigned int width = imageSize[0];
        unsigned int height = imageSize[1];

        utils::writePNGImage(imageBuffer.data(), width, height, outputFile);
    });

    // Process left image on GPU
    make_edge(fileReaderOne, splitArgumentsLeftNode);
    make_edge(output_port<0>(splitArgumentsLeftNode), input_port<0>(leftImageEffect));
    make_edge(output_port<1>(splitArgumentsLeftNode), input_port<1>(leftImageEffect));

    // Pass OpenCL NDRange via input port because it depends on input data
    make_edge(output_port<2>(splitArgumentsLeftNode), input_port<2>(leftImageEffect));

    // Process right image on CPU
    make_edge(fileReaderTwo, rightImageEffect);

    // Merge images on GPU
    make_edge(output_port<0>(leftImageEffect), input_port<0>(mergeImages));
    make_edge(rightImageEffect, input_port<1>(mergeImages));
    make_edge(output_port<1>(leftImageEffect), input_port<2>(mergeImages));

    // Pass OpenCL NDRange via input port because it depends on input data
    make_edge(output_port<2>(leftImageEffect), input_port<3>(mergeImages));

    // Write image to PNG
    make_edge(output_port<0>(mergeImages), input_port<0>(joinTupleNode));
    make_edge(output_port<3>(mergeImages), input_port<1>(joinTupleNode));
    make_edge(joinTupleNode, outputWriter);

    // Define where to get ndrange and kernel arguments
    leftImageEffect.set_args(port_ref<0, 1>());
    leftImageEffect.set_range(port_ref<2>());

    mergeImages.set_args(port_ref<0, 2>());
    mergeImages.set_range(port_ref<3>());

    // Start graph image processing pipeline
    fileReaderOne.try_put(firstFile);
    fileReaderTwo.try_put(secondFile);

    g.wait_for_all();
}


int main(int argc, char* argv[]) {
    try {

        tbb::tick_count mainStartTime = tbb::tick_count::now();

        bool verbose = false;
        std::string algVersion;

        std::string inputFileFirst;
        std::string inputFileSecond;
        std::string outputFile = "output.png";

        utility::parse_cli_arguments(argc, argv,
            utility::cli_argument_pack()
            //"-h" option for displaying help
            .arg(verbose, "-v", "verbose mode")
            .arg(algVersion, "-alg", "name of the used pipeline realisation - can be host, target (default) or host_target")
            .positional_arg(inputFileFirst, "first_filename", "first input file name")
            .positional_arg(inputFileSecond, "second_filename", "second input file name")
        );

        if (!utils::isBothImagesExists(inputFileFirst, inputFileSecond)) {
            std::cout << "Info: one or both images does not exists or empty. Input images will be generated instead." << std::endl;
            inputFileFirst.clear();
            inputFileSecond.clear();
        } else {
            std::cout << "First input file name: " << inputFileFirst << std::endl;
            std::cout << "Second input file name: " << inputFileSecond << std::endl;
        }

        if (algVersion.empty() || algVersion == "target") {
            openclFunctionGPU(inputFileFirst, inputFileSecond, outputFile);
        } else if (algVersion == "host_target") {
            openclFunctionGPUPlusCPU(inputFileFirst, inputFileSecond, outputFile);
        } else if (algVersion == "host") {
            hostFunction(inputFileFirst, inputFileSecond, outputFile);
        }

        utility::report_elapsed_time((tbb::tick_count::now() - mainStartTime).seconds());

        return 0;
    } catch (std::exception& e) {
        std::cerr << "Error occurred :\"" << e.what() << "\"\n";
        return -1;
    }
}
#else
int main() {
    utility::report_skipped();
    return 0;
}
#endif /* __TBB_PREVIEW_OPENCL_NODE && __TBB_CPP11_LAMBDAS_PRESENT */
