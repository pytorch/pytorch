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

#define TBB_PREVIEW_FLOW_GRAPH_FEATURES 1
#include "tbb/tbb_config.h"
#include "../../common/utility/utility.h"

#if __TBB_PREVIEW_ASYNC_MSG && __TBB_CPP11_LAMBDAS_PRESENT

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <queue>

#include "bzlib.h"

#include "tbb/flow_graph.h"
#include "tbb/tick_count.h"
#include "tbb/compat/thread"
#include "tbb/concurrent_queue.h"

// TODO: change memory allocation/deallocation to be managed in constructor/destructor
struct Buffer {
    size_t len;
    char* b;
};

struct BufferMsg {

    BufferMsg() {}
    BufferMsg(Buffer& inputBuffer, Buffer& outputBuffer, size_t seqId, bool isLast = false)
        : inputBuffer(inputBuffer), outputBuffer(outputBuffer), seqId(seqId), isLast(isLast) {}

    static BufferMsg createBufferMsg(size_t seqId, size_t chunkSize) {
        Buffer inputBuffer;
        inputBuffer.b = new char[chunkSize];
        inputBuffer.len = chunkSize;

        Buffer outputBuffer;
        size_t compressedChunkSize = chunkSize * 1.01 + 600; // compression overhead
        outputBuffer.b = new char[compressedChunkSize];
        outputBuffer.len = compressedChunkSize;

        return BufferMsg(inputBuffer, outputBuffer, seqId);
    }

    static void destroyBufferMsg(const BufferMsg& destroyMsg) {
        delete[] destroyMsg.inputBuffer.b;
        delete[] destroyMsg.outputBuffer.b;
    }

    void markLast(size_t lastId) {
        isLast = true;
        seqId = lastId;
    }

    size_t seqId;
    Buffer inputBuffer;
    Buffer outputBuffer;
    bool isLast;
};

class BufferCompressor {
public:

    BufferCompressor(int blockSizeIn100KB) : m_blockSize(blockSizeIn100KB) {}

    BufferMsg operator()(BufferMsg buffer) const {
        if (!buffer.isLast) {
            unsigned int outSize = buffer.outputBuffer.len;
            BZ2_bzBuffToBuffCompress(buffer.outputBuffer.b, &outSize,
                buffer.inputBuffer.b, buffer.inputBuffer.len,
                m_blockSize, 0, 30);
            buffer.outputBuffer.len = outSize;
        }
        return buffer;
    }

private:
    int m_blockSize;
};

class IOOperations {
public:

    IOOperations(std::ifstream& inputStream, std::ofstream& outputStream, size_t chunkSize)
        : m_inputStream(inputStream), m_outputStream(outputStream), m_chunkSize(chunkSize), m_chunksRead(0) {}

    void readChunk(Buffer& buffer) {
        m_inputStream.read(buffer.b, m_chunkSize);
        buffer.len = static_cast<size_t>(m_inputStream.gcount());
        m_chunksRead++;
    }

    void writeChunk(const Buffer& buffer) {
        m_outputStream.write(buffer.b, buffer.len);
    }

    size_t chunksRead() const {
        return m_chunksRead;
    }

    size_t chunkSize() const {
        return m_chunkSize;
    }

    bool hasDataToRead() const {
        return m_inputStream.is_open() && !m_inputStream.eof();
    }

private:

    std::ifstream& m_inputStream;
    std::ofstream& m_outputStream;

    size_t m_chunkSize;
    size_t m_chunksRead;
};

//-----------------------------------------------------------------------------------------------------------------------
//---------------------------------------Compression example based on async_node-----------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

typedef tbb::flow::async_node< tbb::flow::continue_msg, BufferMsg > async_file_reader_node;
typedef tbb::flow::async_node< BufferMsg, tbb::flow::continue_msg > async_file_writer_node;

class AsyncNodeActivity {
public:

    AsyncNodeActivity(IOOperations& io)
        : m_io(io), m_fileWriterThread(&AsyncNodeActivity::writingLoop, this) {}

    ~AsyncNodeActivity() {
        m_fileReaderThread.join();
        m_fileWriterThread.join();
    }

    void submitRead(async_file_reader_node::gateway_type& gateway) {
        gateway.reserve_wait();
        std::thread(&AsyncNodeActivity::readingLoop, this, std::ref(gateway)).swap(m_fileReaderThread);
    }

    void submitWrite(const BufferMsg& bufferMsg) {
        m_writeQueue.push(bufferMsg);
    }

private:

    void readingLoop(async_file_reader_node::gateway_type& gateway) {
        while (m_io.hasDataToRead()) {
            BufferMsg bufferMsg = BufferMsg::createBufferMsg(m_io.chunksRead(), m_io.chunkSize());
            m_io.readChunk(bufferMsg.inputBuffer);
            gateway.try_put(bufferMsg);
        }
        sendLastMessage(gateway);
        gateway.release_wait();
    }

    void writingLoop() {
        BufferMsg buffer;
        m_writeQueue.pop(buffer);
        while (!buffer.isLast) {
            m_io.writeChunk(buffer.outputBuffer);
            m_writeQueue.pop(buffer);
        }
    }

    void sendLastMessage(async_file_reader_node::gateway_type& gateway) {
        BufferMsg lastMsg;
        lastMsg.markLast(m_io.chunksRead());
        gateway.try_put(lastMsg);
    }

    IOOperations& m_io;

    tbb::concurrent_bounded_queue< BufferMsg > m_writeQueue;

    std::thread m_fileReaderThread;
    std::thread m_fileWriterThread;
};

void fgCompressionAsyncNode(IOOperations& io, int blockSizeIn100KB) {
    tbb::flow::graph g;

    AsyncNodeActivity asyncNodeActivity(io);

    async_file_reader_node file_reader(g, tbb::flow::unlimited, [&asyncNodeActivity](const tbb::flow::continue_msg& msg, async_file_reader_node::gateway_type& gateway) {
        asyncNodeActivity.submitRead(gateway);
    });

    tbb::flow::function_node< BufferMsg, BufferMsg > compressor(g, tbb::flow::unlimited, BufferCompressor(blockSizeIn100KB));

    tbb::flow::sequencer_node< BufferMsg > ordering(g, [](const BufferMsg& bufferMsg)->size_t {
        return bufferMsg.seqId;
    });

    // The node is serial to preserve the right order of buffers set by the preceding sequencer_node
    async_file_writer_node output_writer(g, tbb::flow::serial, [&asyncNodeActivity](const BufferMsg& bufferMsg, async_file_writer_node::gateway_type& gateway) {
        asyncNodeActivity.submitWrite(bufferMsg);
    });

    make_edge(file_reader, compressor);
    make_edge(compressor, ordering);
    make_edge(ordering, output_writer);

    file_reader.try_put(tbb::flow::continue_msg());

    g.wait_for_all();
}

//-----------------------------------------------------------------------------------------------------------------------
//------------------------------------------Compression example based on async_msg---------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

typedef tbb::flow::async_msg< BufferMsg > async_msg_type;

class AsyncMsgActivity {
public:

    AsyncMsgActivity(tbb::flow::graph& g, IOOperations& io)
        : m_io(io), m_graph(g), m_fileReaderThread(&AsyncMsgActivity::readingLoop, this),
          m_fileWriterThread(&AsyncMsgActivity::writingLoop, this)
    {
        // Graph synchronization starts here and ends
        // when the last buffer was written in "writing thread"
        m_graph.increment_wait_count();
    }

    ~AsyncMsgActivity() {
        m_fileReaderThread.join();
        m_fileWriterThread.join();

        // Lets release resources that async
        // activity and graph were acquired
        freeBuffers();
    }

    async_msg_type submitRead(BufferMsg& bufferMsg) {
        async_msg_type msg;
        work_type readWork = { bufferMsg, msg };
        m_readQueue.push(readWork);
        return msg;
    }

    async_msg_type submitWrite(const BufferMsg& bufferMsg) {
        async_msg_type msg;
        work_type writeWork = { bufferMsg, msg };
        m_writeQueue.push(writeWork);
        return msg;
    }

private:

    struct work_type {
        BufferMsg bufferMsg;
        async_msg_type msg;
    };

    void readingLoop() {
        work_type readWork;
        m_readQueue.pop(readWork);

        // Reading thread waits for buffers to be received
        // (the graph reuses limitted number of buffers)
        // and reads the file while there is something to read
        while (m_io.hasDataToRead()) {
            readWork.bufferMsg.seqId = m_io.chunksRead();
            m_io.readChunk(readWork.bufferMsg.inputBuffer);
            readWork.msg.set(readWork.bufferMsg);
            m_readQueue.pop(readWork);
        }

        // Pass message with an end flag to the graph
        sendLastMessage(readWork);
    }

    void sendLastMessage(work_type& work) {
        work.bufferMsg.markLast(m_io.chunksRead());
        work.msg.set(work.bufferMsg);
    }

    void writingLoop() {
        work_type writeWork;
        m_writeQueue.pop(writeWork);

        // Writing thread writes all buffers that it gets
        // and reuses them. At the end all reusing buffers
        // is stored in read queue
        while (!writeWork.bufferMsg.isLast) {
            m_io.writeChunk(writeWork.bufferMsg.outputBuffer);
            writeWork.msg.set(writeWork.bufferMsg);
            m_writeQueue.pop(writeWork);
        }

        // Store last message to the reading queue to free resources later
        writeWork.msg.set(writeWork.bufferMsg);

        // After all buffers have been written
        // the synchronization ends
        m_graph.decrement_wait_count();
    }

    void freeBuffers() {
        int buffersNumber = m_readQueue.size();
        for (int i = 0; i < buffersNumber; i++) {
            work_type workToDelete;
            m_readQueue.pop(workToDelete);
            BufferMsg::destroyBufferMsg(workToDelete.bufferMsg);
        }
    }

    IOOperations& m_io;

    tbb::flow::graph& m_graph;

    tbb::concurrent_bounded_queue< work_type > m_writeQueue;
    tbb::concurrent_bounded_queue< work_type > m_readQueue;

    std::thread m_fileReaderThread;
    std::thread m_fileWriterThread;
};

void fgCompressionAsyncMsg(IOOperations& io, int blockSizeIn100KB, size_t memoryLimitIn1MB) {
    // Memory limit sets the number of buffers that can be reused
    int buffersNumber = memoryLimitIn1MB * 1000 * 1024 / io.chunkSize();

    tbb::flow::graph g;

    AsyncMsgActivity asyncMsgActivity(g, io);

    tbb::flow::function_node< BufferMsg, async_msg_type > file_reader(g, tbb::flow::unlimited, [&asyncMsgActivity](BufferMsg bufferMsg) -> async_msg_type {
        return asyncMsgActivity.submitRead(bufferMsg);
    });

    tbb::flow::function_node< BufferMsg, BufferMsg > compressor(g, tbb::flow::unlimited, BufferCompressor(blockSizeIn100KB));

    tbb::flow::sequencer_node< BufferMsg > ordering(g, [](const BufferMsg& bufferMsg) -> size_t {
        return bufferMsg.seqId;
    });

    // The node is serial to preserve the right order of buffers set by the preceding sequencer_node
    tbb::flow::function_node< BufferMsg, async_msg_type > output_writer(g, tbb::flow::serial, [&asyncMsgActivity](const BufferMsg& bufferMsg) -> async_msg_type {
        return asyncMsgActivity.submitWrite(bufferMsg);
    });

    make_edge(file_reader, compressor);
    make_edge(compressor, ordering);
    make_edge(ordering, output_writer);
    make_edge(output_writer, file_reader);

    // Creating buffers to be reused in read/compress/write graph loop
    for (int i = 0; i < buffersNumber; i++) {
        BufferMsg reuseBufferMsg = BufferMsg::createBufferMsg(0, io.chunkSize());
        file_reader.try_put(reuseBufferMsg);
    }

    g.wait_for_all();
}

//-----------------------------------------------------------------------------------------------------------------------
//---------------------------------------------Simple compression example------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

void fgCompression(IOOperations& io, int blockSizeIn100KB) {
    tbb::flow::graph g;

    tbb::flow::source_node< BufferMsg > file_reader(g, [&io](BufferMsg& bufferMsg)->bool {
        if (io.hasDataToRead()) {
            bufferMsg = BufferMsg::createBufferMsg(io.chunksRead(), io.chunkSize());
            io.readChunk(bufferMsg.inputBuffer);
            return true;
        }
        return false;
    });

    tbb::flow::function_node< BufferMsg, BufferMsg > compressor(g, tbb::flow::unlimited, BufferCompressor(blockSizeIn100KB));

    tbb::flow::sequencer_node< BufferMsg > ordering(g, [](const BufferMsg& buffer)->size_t {
        return buffer.seqId;
    });

    tbb::flow::function_node< BufferMsg > output_writer(g, tbb::flow::serial, [&io](const BufferMsg& bufferMsg) {
        io.writeChunk(bufferMsg.outputBuffer);
        BufferMsg::destroyBufferMsg(bufferMsg);
    });

    make_edge(file_reader, compressor);
    make_edge(compressor, ordering);
    make_edge(ordering, output_writer);

    g.wait_for_all();
}

//-----------------------------------------------------------------------------------------------------------------------

bool endsWith(const std::string& str, const std::string& suffix) {
    return str.find(suffix, str.length() - suffix.length()) != std::string::npos;
}

//-----------------------------------------------------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    try {
        tbb::tick_count mainStartTime = tbb::tick_count::now();

        const std::string archiveExtension = ".bz2";
        bool verbose = false;
        std::string asyncType;
        std::string inputFileName;
        int blockSizeIn100KB = 1; // block size in 100KB chunks
        size_t memoryLimitIn1MB = 1; // memory limit for compression in megabytes granularity

        utility::parse_cli_arguments(argc, argv,
            utility::cli_argument_pack()
            //"-h" option for displaying help is present implicitly
            .arg(blockSizeIn100KB, "-b", "\t block size in 100KB chunks, [1 .. 9]")
            .arg(verbose, "-v", "verbose mode")
            .arg(memoryLimitIn1MB, "-l", "used memory limit for compression algorithm in 1MB (minimum) granularity")
            .arg(asyncType, "-a", "name of the used graph async implementation - can be async_node or async_msg")
            .positional_arg(inputFileName, "filename", "input file name")
        );

        if (inputFileName.empty()) {
            throw std::invalid_argument("Input file name is not specified. Try 'fgbzip2 -h' for more information.");
        }

        if (blockSizeIn100KB < 1 || blockSizeIn100KB > 9) {
            throw std::invalid_argument("Incorrect block size. Try 'fgbzip2 -h' for more information.");
        }

        if (memoryLimitIn1MB < 1) {
            throw std::invalid_argument("Incorrect memory limit size. Try 'fgbzip2 -h' for more information.");
        }

        if (verbose) std::cout << "Input file name: " << inputFileName << std::endl;
        if (endsWith(inputFileName, archiveExtension)) {
            throw std::invalid_argument("Input file already have " + archiveExtension + " extension.");
        }

        std::ifstream inputStream(inputFileName.c_str(), std::ios::in | std::ios::binary);
        if (!inputStream.is_open()) {
            throw std::invalid_argument("Cannot open " + inputFileName + " file.");
        }

        std::string outputFileName(inputFileName + archiveExtension);

        std::ofstream outputStream(outputFileName.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
        if (!outputStream.is_open()) {
            throw std::invalid_argument("Cannot open " + outputFileName + " file.");
        }

        // General interface to work with I/O buffers operations
        size_t chunkSize = blockSizeIn100KB * 100 * 1024;
        IOOperations io(inputStream, outputStream, chunkSize);

        if (asyncType.empty()) {
            if (verbose) std::cout << "Running flow graph based compression algorithm." << std::endl;
            fgCompression(io, blockSizeIn100KB);
        } else if (asyncType == "async_node") {
            if (verbose) std::cout << "Running flow graph based compression algorithm with async_node based asynchronious IO operations." << std::endl;
            fgCompressionAsyncNode(io, blockSizeIn100KB);
        } else if (asyncType == "async_msg") {
            if (verbose) std::cout << "Running flow graph based compression algorithm with async_msg based asynchronious IO operations. Using limited memory: " << memoryLimitIn1MB << "MB." << std::endl;
            fgCompressionAsyncMsg(io, blockSizeIn100KB, memoryLimitIn1MB);
        }

        inputStream.close();
        outputStream.close();

        utility::report_elapsed_time((tbb::tick_count::now() - mainStartTime).seconds());

        return 0;
    } catch (std::exception& e) {
        std::cerr << "Error occurred. Error text is : \"" << e.what() << "\"\n";
        return -1;
    }
}
#else
int main() {
    utility::report_skipped();
    return 0;
}
#endif /* __TBB_PREVIEW_ASYNC_NODE && __TBB_CPP11_LAMBDAS_PRESENT */
