#ifndef CAFFE2_VIDEO_VIDEO_DECODER_H_
#define CAFFE2_VIDEO_VIDEO_DECODER_H_

#include <caffe2/core/logging.h>
#include <stdio.h>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
}

namespace caffe2 {

#define VIO_BUFFER_SZ 32768
#define MAX_DECODING_FRAMES 10000

// enum to specify 3 special fps sampling behaviors:
// 0: disable fps sampling, no frame sampled at all
// -1: unlimited fps sampling, will sample at native video fps
// -2: disable fps sampling, but will get the frame at specific timestamp
enum SpecialFps {
  SAMPLE_NO_FRAME = 0,
  SAMPLE_ALL_FRAMES = -1,
  SAMPLE_TIMESTAMP_ONLY = -2,
};

// three different types of resolution when decoding the video
// 0: resize to width x height and ignore the aspect ratio;
// 1: resize to make size at least (width x height) and keep the aspect ratio;
// 2: using the original resolution of the video; if resolution
//    is smaller than crop_height x crop_width, resize to ensure
//    new height >= crop_height and new width >= crop_width
//    and keep the aspect ratio;
enum VideoResType {
  USE_WIDTH_HEIGHT = 0,
  USE_MINIMAL_WIDTH_HEIGHT = 1,
  ORIGINAL_RES = 2,
};

// three different types of decoding behavior are supported
// 0: do temporal jittering to sample a random clip from the video
// 1: sample a clip from a given starting frame
// 2: uniformly sample multiple clips from the video;
enum DecodeType {
  DO_TMP_JITTER = 0,
  DO_UNIFORM_SMP = 1,
  USE_START_FRM = 2,
};

// sampling interval for fps starting at specified timestamp
// use enum SpecialFps to set special fps decoding behavior
// note sampled fps will not always accurately follow the target fps,
// because sampled frame has to snap to actual frame timestamp,
// e.g. video fps = 25, sample fps = 4 will sample every 0.28s, not 0.25
// video fps = 25, sample fps = 5 will sample every 0.24s, not 0.2,
// because of floating-point division accuracy (1 / 5.0 is not exactly 0.2)
struct SampleInterval {
  double timestamp;
  double fps;
  SampleInterval() : timestamp(-1), fps(SpecialFps::SAMPLE_ALL_FRAMES) {}
  SampleInterval(double ts, double f) : timestamp(ts), fps(f) {}
  bool operator<(const SampleInterval& itvl) const {
    return (timestamp < itvl.timestamp);
  }
};

class Params {
 public:
  // return all key-frames regardless of specified fps
  bool keyFrames_ = false;

  // Output image pixel format
  AVPixelFormat pixelFormat_ = AVPixelFormat::AV_PIX_FMT_RGB24;

  // Index of stream to decode.
  // -1 will automatically decode the first video stream.
  int streamIndex_ = -1;

  // How many frames to output at most from the video
  // -1 no limit
  int maximumOutputFrames_ = -1;

  // params for video resolution
  int video_res_type_ = VideoResType::USE_WIDTH_HEIGHT;

  // the size of the patch croped from the input video
  int crop_height_ = -1;
  int crop_width_ = -1;

  // minimal resolution for resizing when using USE_MINIMAL_WIDTH_HEIGHT
  int height_min_ = -1;
  int width_min_ = -1;

  // the video resolution after resizing
  int scale_w_ = -1;
  int scale_h_ = -1;

  // params for decoding behavior
  int decode_type_ = DecodeType::DO_TMP_JITTER;
  int num_of_required_frame_ = -1;

  // intervals_ control variable sampling fps between different timestamps
  // intervals_ must be ordered strictly ascending by timestamps
  // the first interval must have a timestamp of zero
  // fps must be either the 3 special fps defined in SpecialFps, or > 0
  std::vector<SampleInterval> intervals_ = {{0, SpecialFps::SAMPLE_ALL_FRAMES}};

  Params() {}

  /**
   * FPS of output frames
   * setting here will reset intervals_ and force decoding at target FPS
   * This can be used if user just want to decode at a steady fps
   */
  Params& fps(float v) {
    intervals_.clear();
    intervals_.emplace_back(0, v);
    return *this;
  }

  /**
   * Sample output frames at a specified list of timestamps
   * Timestamps must be in increasing order, and timestamps past the end of the
   * video will be ignored
   * Setting here will reset intervals_
   */
  Params& setSampleTimestamps(const std::vector<double>& timestamps) {
    intervals_.clear();
    // insert an interval per desired frame.
    for (auto& timestamp : timestamps) {
      intervals_.emplace_back(timestamp, SpecialFps::SAMPLE_TIMESTAMP_ONLY);
    }
    return *this;
  }

  /**
   * Pixel format of output buffer, default PIX_FMT_RGB24
   */
  Params& pixelFormat(AVPixelFormat pixelFormat) {
    pixelFormat_ = pixelFormat;
    return *this;
  }

  /**
   * Return all key-frames
   */
  Params& keyFrames(bool keyFrames) {
    keyFrames_ = keyFrames;
    return *this;
  }

  /**
   * Index of video stream to process, defaults to the first video stream
   */
  Params& streamIndex(int index) {
    streamIndex_ = index;
    return *this;
  }

  /**
   * Only output this many frames, default to no limit
   */
  Params& maxOutputFrames(int count) {
    maximumOutputFrames_ = count;
    return *this;
  }

  /**
   * Output frame width, default to video width
   */
  Params& outputWidth(int width) {
    scale_w_ = width;
    return *this;
  }

  /**
   * Output frame height, default to video height
   */
  Params& outputHeight(int height) {
    scale_h_ = height;
    return *this;
  }
};

// data structure for storing decoded video frames
class DecodedFrame {
 public:
  struct avDeleter {
    void operator()(unsigned char* p) const {
      av_free(p);
    }
  };
  using AvDataPtr = std::unique_ptr<uint8_t, avDeleter>;

  // decoded data buffer
  AvDataPtr data_;

  // size in bytes
  int size_ = 0;

  // frame dimensions
  int width_ = 0;
  int height_ = 0;

  // timestamp in seconds since beginning of video
  double timestamp_ = 0;

  // true if this is a key frame.
  bool keyFrame_ = false;

  // index of frame in video
  int index_ = -1;

  // Sequential number of outputted frame
  int outputFrameIndex_ = -1;
};

class VideoIOContext {
 public:
  explicit VideoIOContext(const std::string& fname)
      : workBuffersize_(VIO_BUFFER_SZ),
        workBuffer_((uint8_t*)av_malloc(workBuffersize_)),
        inputFile_(nullptr),
        inputBuffer_(nullptr),
        inputBufferSize_(0) {
    inputFile_ = fopen(fname.c_str(), "rb");
    if (inputFile_ == nullptr) {
      LOG(ERROR) << "Error opening video file " << fname;
    }
    ctx_ = avio_alloc_context(
        static_cast<unsigned char*>(workBuffer_.get()),
        workBuffersize_,
        0,
        this,
        &VideoIOContext::readFile,
        nullptr, // no write function
        &VideoIOContext::seekFile);
  }

  explicit VideoIOContext(const char* buffer, int size)
      : workBuffersize_(VIO_BUFFER_SZ),
        workBuffer_((uint8_t*)av_malloc(workBuffersize_)),
        inputFile_(nullptr),
        inputBuffer_(buffer),
        inputBufferSize_(size) {
    ctx_ = avio_alloc_context(
        static_cast<unsigned char*>(workBuffer_.get()),
        workBuffersize_,
        0,
        this,
        &VideoIOContext::readMemory,
        nullptr, // no write function
        &VideoIOContext::seekMemory);
  }

  ~VideoIOContext() {
    av_free(ctx_);
    if (inputFile_) {
      fclose(inputFile_);
    }
  }

  int read(unsigned char* buf, int buf_size) {
    if (inputBuffer_) {
      return readMemory(this, buf, buf_size);
    } else if (inputFile_) {
      return readFile(this, buf, buf_size);
    } else {
      return -1;
    }
  }

  int64_t seek(int64_t offset, int whence) {
    if (inputBuffer_) {
      return seekMemory(this, offset, whence);
    } else if (inputFile_) {
      return seekFile(this, offset, whence);
    } else {
      return -1;
    }
  }

  static int readFile(void* opaque, unsigned char* buf, int buf_size) {
    VideoIOContext* h = static_cast<VideoIOContext*>(opaque);
    if (feof(h->inputFile_)) {
      return AVERROR_EOF;
    }
    size_t ret = fread(buf, 1, buf_size, h->inputFile_);
    if (ret < buf_size) {
      if (ferror(h->inputFile_)) {
        return -1;
      }
    }
    return ret;
  }

  static int64_t seekFile(void* opaque, int64_t offset, int whence) {
    VideoIOContext* h = static_cast<VideoIOContext*>(opaque);
    switch (whence) {
      case SEEK_CUR: // from current position
      case SEEK_END: // from eof
      case SEEK_SET: // from beginning of file
        return fseek(h->inputFile_, static_cast<long>(offset), whence);
        break;
      case AVSEEK_SIZE:
        int64_t cur = ftell(h->inputFile_);
        fseek(h->inputFile_, 0L, SEEK_END);
        int64_t size = ftell(h->inputFile_);
        fseek(h->inputFile_, cur, SEEK_SET);
        return size;
    }

    return -1;
  }

  static int readMemory(void* opaque, unsigned char* buf, int buf_size) {
    VideoIOContext* h = static_cast<VideoIOContext*>(opaque);
    if (buf_size < 0) {
      return -1;
    }

    int reminder = h->inputBufferSize_ - h->offset_;
    int r = buf_size < reminder ? buf_size : reminder;
    if (r < 0) {
      return AVERROR_EOF;
    }

    memcpy(buf, h->inputBuffer_ + h->offset_, r);
    h->offset_ += r;
    return r;
  }

  static int64_t seekMemory(void* opaque, int64_t offset, int whence) {
    VideoIOContext* h = static_cast<VideoIOContext*>(opaque);
    switch (whence) {
      case SEEK_CUR: // from current position
        h->offset_ += offset;
        break;
      case SEEK_END: // from eof
        h->offset_ = h->inputBufferSize_ + offset;
        break;
      case SEEK_SET: // from beginning of file
        h->offset_ = offset;
        break;
      case AVSEEK_SIZE:
        return h->inputBufferSize_;
    }
    return h->offset_;
  }

  AVIOContext* get_avio() {
    return ctx_;
  }

 private:
  int workBuffersize_;
  DecodedFrame::AvDataPtr workBuffer_;
  // for file mode
  FILE* inputFile_;

  // for memory mode
  const char* inputBuffer_;
  int inputBufferSize_;
  int offset_ = 0;

  AVIOContext* ctx_;
};

struct VideoMeta {
  double fps;
  int width;
  int height;
  enum AVMediaType codec_type;
  AVPixelFormat pixFormat;
  VideoMeta()
      : fps(-1),
        width(-1),
        height(-1),
        codec_type(AVMEDIA_TYPE_VIDEO),
        pixFormat(AVPixelFormat::AV_PIX_FMT_RGB24) {}
};

class VideoDecoder {
 public:
  VideoDecoder();

  void decodeFile(
      const std::string& filename,
      const Params& params,
      const int start_frm,
      std::vector<std::unique_ptr<DecodedFrame>>& sampledFrames);

  void decodeMemory(
      const char* buffer,
      const int size,
      const Params& params,
      const int start_frm,
      std::vector<std::unique_ptr<DecodedFrame>>& sampledFrames);

 private:
  std::string ffmpegErrorStr(int result);

  void ResizeAndKeepAspectRatio(
      const int origHeight,
      const int origWidth,
      const int heightMin,
      const int widthMin,
      int& outHeight,
      int& outWidth);

  void decodeLoop(
      const std::string& videoName,
      VideoIOContext& ioctx,
      const Params& params,
      const int start_frm,
      std::vector<std::unique_ptr<DecodedFrame>>& sampledFrames);
};
} // namespace caffe2

#endif // CAFFE2_VIDEO_VIDEO_DECODER_H_
