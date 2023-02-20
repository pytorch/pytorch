#include <assert.h>
#include <caffe2/core/logging.h>
#include <caffe2/video/video_decoder.h>
#include <array>
#include <mutex>
#include <random>

namespace caffe2 {

VideoDecoder::VideoDecoder() {
  static bool gInitialized = false;
  static std::mutex gMutex;
  std::unique_lock<std::mutex> lock(gMutex);
  if (!gInitialized) {
    av_register_all();
    avcodec_register_all();
    avformat_network_init();
    gInitialized = true;
  }
}

void VideoDecoder::getAudioSample(
    AVPacket& packet,
    AVCodecContext* audioCodecContext_,
    AVFrame* audioStreamFrame_,
    SwrContext* convertCtx_,
    Callback& callback,
    const Params& params) {
  int frame_finished = 0;
  auto result = avcodec_decode_audio4(
      audioCodecContext_, audioStreamFrame_, &frame_finished, &packet);

  if (frame_finished) {
    // from
    // https://www.ffmpeg.org/doxygen/2.3/decoding_encoding_8c-example.html#a57
    auto c = audioCodecContext_;
    int data_size = av_samples_get_buffer_size(
        nullptr, c->channels, audioStreamFrame_->nb_samples, c->sample_fmt, 1);
    if (data_size < 0) {
      // This should not occur, checking just for paranoia
      LOG(ERROR) << "Failed to calculate data size";
    }

    // from https://www.ffmpeg.org/doxygen/2.1/group__lswr.html#details
    uint8_t* output;
    auto swr = convertCtx_;
    auto inrate = audioCodecContext_->sample_rate;
    auto in_samples = audioStreamFrame_->nb_samples;

    int out_samples = av_rescale_rnd(
        swr_get_delay(swr, inrate) + in_samples,
        params.outrate_,
        inrate,
        AV_ROUND_UP);

    if (out_samples > 0) {
      auto input = (const uint8_t**)&audioStreamFrame_->data[0];
      av_samples_alloc(
          &output,
          nullptr,
          c->channels,
          out_samples,
          (AVSampleFormat)params.outfmt_,
          0);

      // resample the audio data
      out_samples = swr_convert(swr, &output, out_samples, input, in_samples);
      auto sample_size = out_samples * c->channels * sizeof(float);
      auto buffer = std::make_unique<float[]>(sample_size);
      memcpy(buffer.get(), output, sample_size);
      av_freep(&output);

      unique_ptr<DecodedAudio> audio_sample = make_unique<DecodedAudio>();
      audio_sample->dataSize_ = data_size;
      audio_sample->outSampleSize_ = out_samples * c->channels;
      audio_sample->audio_data_ = std::move(buffer);
      callback.audioDecoded(std::move(audio_sample));
    }
  } else {
    result = packet.size;
  }
  packet.size -= result;
  packet.data += result;
}

void VideoDecoder::ResizeAndKeepAspectRatio(
    const int origWidth,
    const int origHeight,
    const int short_edge,
    const int long_edge,
    int& outWidth,
    int& outHeight) {
  if (origWidth < origHeight) {
    // dominant height
    if (short_edge > 0) {
      // use short_edge for rescale
      float ratio = short_edge / float(origWidth);
      outWidth = short_edge;
      outHeight = (int)round(ratio * origHeight);
    } else {
      // use long_edge for rescale
      float ratio = long_edge / float(origHeight);
      outHeight = long_edge;
      outWidth = (int)round(ratio * origWidth);
    }
  } else {
    // dominant width
    if (short_edge > 0) {
      // use short_edge for rescale
      float ratio = short_edge / float(origHeight);
      outHeight = short_edge;
      outWidth = (int)round(ratio * origWidth);
    } else {
      // use long_edge for rescale
      float ratio = long_edge / float(origWidth);
      outWidth = long_edge;
      outHeight = (int)round(ratio * origHeight);
    }
  }
}

void VideoDecoder::decodeLoop(
    const string& videoName,
    VideoIOContext& ioctx,
    const Params& params,
    const int start_frm,
    Callback& callback) {
  AVPixelFormat pixFormat = params.pixelFormat_;
  AVFormatContext* inputContext = avformat_alloc_context();
  AVStream* videoStream_ = nullptr;
  AVCodecContext* videoCodecContext_ = nullptr;
  AVCodecContext* audioCodecContext_ = nullptr;
  AVFrame* videoStreamFrame_ = nullptr;
  AVFrame* audioStreamFrame_ = nullptr;
  SwrContext* convertCtx_ = nullptr;
  AVPacket packet;
  av_init_packet(&packet); // init packet
  SwsContext* scaleContext_ = nullptr;

  try {
    inputContext->pb = ioctx.get_avio();
    inputContext->flags |= AVFMT_FLAG_CUSTOM_IO;
    int ret = 0;

    // Determining the input format:
    int probeSz = 1 * 1024 + AVPROBE_PADDING_SIZE;
    DecodedFrame::AvDataPtr probe((uint8_t*)av_malloc(probeSz));
    memset(probe.get(), 0, probeSz);
    int len = ioctx.read(probe.get(), probeSz - AVPROBE_PADDING_SIZE);
    if (len < probeSz - AVPROBE_PADDING_SIZE) {
      LOG(ERROR) << "Insufficient data to determine video format";
      return;
    }
    // seek back to start of stream
    ioctx.seek(0, SEEK_SET);

    unique_ptr<AVProbeData> probeData(new AVProbeData());
    probeData->buf = probe.get();
    probeData->buf_size = len;
    probeData->filename = "";
    // Determine the input-format:
    inputContext->iformat = av_probe_input_format(probeData.get(), 1);
    // this is to avoid the double-free error
    if (inputContext->iformat == nullptr) {
      LOG(ERROR) << "inputContext iformat is nullptr!";
      return;
    }

    ret = avformat_open_input(&inputContext, "", nullptr, nullptr);
    if (ret < 0) {
      LOG(ERROR) << "Unable to open stream : " << ffmpegErrorStr(ret);
      return;
    }

    ret = avformat_find_stream_info(inputContext, nullptr);
    if (ret < 0) {
      LOG(ERROR) << "Unable to find stream info in " << videoName << " "
                 << ffmpegErrorStr(ret);
      return;
    }

    // Decode the first video stream
    int videoStreamIndex_ = params.streamIndex_;
    int audioStreamIndex_ = params.streamIndex_;
    if (params.streamIndex_ == -1) {
      for (int i = 0; i < inputContext->nb_streams; i++) {
        auto stream = inputContext->streams[i];
        if (stream->codec->codec_type == AVMEDIA_TYPE_VIDEO &&
            videoStreamIndex_ == -1) {
          videoStreamIndex_ = i;
          videoStream_ = stream;
        } else if (
            stream->codec->codec_type == AVMEDIA_TYPE_AUDIO &&
            audioStreamIndex_ == -1) {
          audioStreamIndex_ = i;
        }
        if (videoStreamIndex_ != -1 && audioStreamIndex_ != -1) {
          break;
        }
      }
    }
    if (videoStream_ == nullptr) {
      LOG(ERROR) << "Unable to find video stream in " << videoName << " "
                 << ffmpegErrorStr(ret);
      return;
    }

    // Initialize codec
    AVDictionary* opts = nullptr;
    videoCodecContext_ = videoStream_->codec;
    try {
      ret = avcodec_open2(
          videoCodecContext_,
          avcodec_find_decoder(videoCodecContext_->codec_id),
          &opts);
    } catch (const std::exception&) {
      LOG(ERROR) << "Exception during open video codec";
      return;
    }

    if (ret < 0) {
      LOG(ERROR) << "Cannot open video codec : "
                 << videoCodecContext_->codec->name;
      return;
    }

    if (params.getAudio_ && audioStreamIndex_ >= 0) {
      // see e.g. ridge/decoder/StreamDecoder.cpp
      audioCodecContext_ = inputContext->streams[audioStreamIndex_]->codec;
      ret = avcodec_open2(
          audioCodecContext_,
          avcodec_find_decoder(audioCodecContext_->codec_id),
          nullptr);

      if (ret < 0) {
        LOG(ERROR) << "Cannot open audio codec : "
                   << audioCodecContext_->codec->name;
        return;
      }

      convertCtx_ = swr_alloc_set_opts(
          nullptr,
          params.outlayout_,
          (AVSampleFormat)params.outfmt_,
          params.outrate_,
          audioCodecContext_->channel_layout,
          audioCodecContext_->sample_fmt,
          audioCodecContext_->sample_rate,
          0,
          nullptr);

      if (convertCtx_ == nullptr) {
        LOG(ERROR) << "Cannot setup sample format converter.";
        return;
      }
      if (swr_init(convertCtx_) < 0) {
        LOG(ERROR) << "Cannot init sample format converter.";
        return;
      }
    }

    // Calculate if we need to rescale the frames
    const int origWidth = videoCodecContext_->width;
    const int origHeight = videoCodecContext_->height;
    int outWidth = origWidth;
    int outHeight = origHeight;

    if (params.video_res_type_ == VideoResType::ORIGINAL_RES) {
      // if the original resolution is too low,
      // make it at least the same size as crop_size_
      if (params.crop_size_ > origWidth || params.crop_size_ > origHeight) {
        ResizeAndKeepAspectRatio(
            origWidth, origHeight, params.crop_size_, -1, outWidth, outHeight);
      }
    } else if (params.video_res_type_ == VideoResType::USE_SHORT_EDGE) {
      // resize the image to the predefined
      // short_edge_ resolution while keep the aspect ratio
      ResizeAndKeepAspectRatio(
          origWidth, origHeight, params.short_edge_, -1, outWidth, outHeight);
    } else if (params.video_res_type_ == VideoResType::USE_WIDTH_HEIGHT) {
      // resize the image to the predefined
      // resolution and ignore the aspect ratio
      outWidth = params.outputWidth_;
      outHeight = params.outputHeight_;
    } else {
      LOG(ERROR) << "Unknown VideoResType: " << params.video_res_type_;
      return;
    }

    // Make sure that we have a valid format
    if (videoCodecContext_->pix_fmt == AV_PIX_FMT_NONE) {
      LOG(ERROR) << "pixel format is not valid.";
      return;
    }

    // Create a scale context
    scaleContext_ = sws_getContext(
        videoCodecContext_->width,
        videoCodecContext_->height,
        videoCodecContext_->pix_fmt,
        outWidth,
        outHeight,
        pixFormat,
        SWS_FAST_BILINEAR,
        nullptr,
        nullptr,
        nullptr);

    // Getting video meta data
    VideoMeta videoMeta;
    videoMeta.codec_type = videoCodecContext_->codec_type;
    videoMeta.width = outWidth;
    videoMeta.height = outHeight;
    videoMeta.pixFormat = pixFormat;

    // avoid division by zero, code adapted from
    // https://www.ffmpeg.org/doxygen/0.6/rational_8h-source.html
    if (videoStream_->avg_frame_rate.num == 0 ||
        videoStream_->avg_frame_rate.den == 0) {
      LOG(ERROR) << "Frame rate is wrong. No data found.";
      return;
    }

    videoMeta.fps = av_q2d(videoStream_->avg_frame_rate);
    callback.videoDecodingStarted(videoMeta);

    if (params.intervals_.size() == 0) {
      LOG(ERROR) << "Empty sampling intervals.";
      return;
    }

    std::vector<SampleInterval>::const_iterator itvlIter =
        params.intervals_.begin();
    if (itvlIter->timestamp != 0) {
      LOG(ERROR) << "Sampling interval starting timestamp is not zero.";
      return;
    }

    double currFps = itvlIter->fps;
    if (currFps < 0 && currFps != SpecialFps::SAMPLE_ALL_FRAMES &&
        currFps != SpecialFps::SAMPLE_TIMESTAMP_ONLY) {
      // fps must be 0, -1, -2 or > 0
      LOG(ERROR) << "Invalid sampling fps.";
      return;
    }

    double prevTimestamp = itvlIter->timestamp;
    itvlIter++;
    if (itvlIter != params.intervals_.end() &&
        prevTimestamp >= itvlIter->timestamp) {
      LOG(ERROR) << "Sampling interval timestamps must be strictly ascending.";
      return;
    }

    double lastFrameTimestamp = -1.0;
    double timestamp = -1.0;

    // Initialize frame and packet.
    // These will be reused across calls.
    videoStreamFrame_ = av_frame_alloc();
    audioStreamFrame_ = av_frame_alloc();

    // frame index in video stream
    int frameIndex = -1;
    // frame index of outputed frames
    int outputFrameIndex = -1;

    /* identify the starting point from where we must start decoding */
    std::mt19937 meta_randgen(time(nullptr));
    long int start_ts = -1;
    bool mustDecodeAll = false;

    if (videoStream_->duration > 0 && videoStream_->nb_frames > 0) {
      /* we have a valid duration and nb_frames. We can safely
       * detect an intermediate timestamp to start decoding from. */

      // leave a margin of 10 frames to take in to account the error
      // from av_seek_frame
      long int margin =
          int(ceil((10 * videoStream_->duration) / (videoStream_->nb_frames)));
      // if we need to do temporal jittering
      if (params.decode_type_ == DecodeType::DO_TMP_JITTER) {
        /* estimate the average duration for the required # of frames */
        double maxFramesDuration =
            (videoStream_->duration * params.num_of_required_frame_) /
            (videoStream_->nb_frames);
        int ts1 = 0;
        int ts2 = videoStream_->duration - int(ceil(maxFramesDuration));
        ts2 = ts2 > 0 ? ts2 : 0;
        // pick a random timestamp between ts1 and ts2. ts2 is selected such
        // that you have enough frames to satisfy the required # of frames.
        start_ts = std::uniform_int_distribution<>(ts1, ts2)(meta_randgen);
        // seek a frame at start_ts
        ret = av_seek_frame(
            inputContext,
            videoStreamIndex_,
            0 > (start_ts - margin) ? 0 : (start_ts - margin),
            AVSEEK_FLAG_BACKWARD);

        // if we need to decode from the start_frm
      } else if (params.decode_type_ == DecodeType::USE_START_FRM) {
        if (videoStream_ == nullptr) {
          LOG(ERROR) << "Nullptr found at videoStream_";
          return;
        }
        start_ts = int(floor(
            (videoStream_->duration * start_frm) / (videoStream_->nb_frames)));
        // seek a frame at start_ts
        ret = av_seek_frame(
            inputContext,
            videoStreamIndex_,
            0 > (start_ts - margin) ? 0 : (start_ts - margin),
            AVSEEK_FLAG_BACKWARD);
      } else {
        mustDecodeAll = true;
      }

      if (ret < 0) {
        LOG(INFO) << "Unable to decode from a random start point";
        /* fall back to default decoding of all frames from start */
        av_seek_frame(inputContext, videoStreamIndex_, 0, AVSEEK_FLAG_BACKWARD);
        mustDecodeAll = true;
      }
    } else {
      mustDecodeAll = true;
    }

    int gotPicture = 0;
    int eof = 0;
    int selectiveDecodedFrames = 0;

    int maxFrames = (params.decode_type_ == DecodeType::DO_UNIFORM_SMP)
        ? MAX_DECODING_FRAMES
        : params.num_of_required_frame_;
    // There is a delay between reading packets from the
    // transport and getting decoded frames back.
    // Therefore, after EOF, continue going while
    // the decoder is still giving us frames.
    int ipacket = 0;
    while ((!eof || gotPicture) &&
           /* either you must decode all frames or decode up to maxFrames
            * based on status of the mustDecodeAll flag */
           (mustDecodeAll || (selectiveDecodedFrames < maxFrames)) &&
           /* If on the last interval and not autodecoding keyframes and a
            * SpecialFps indicates no more frames are needed, stop decoding */
           !((itvlIter == params.intervals_.end() &&
              (currFps == SpecialFps::SAMPLE_TIMESTAMP_ONLY ||
               currFps == SpecialFps::SAMPLE_NO_FRAME)) &&
             !params.keyFrames_)) {
      try {
        if (!eof) {
          ret = av_read_frame(inputContext, &packet);
          if (ret == AVERROR_EOF) {
            eof = 1;
            av_free_packet(&packet);
            packet.data = nullptr;
            packet.size = 0;
            // stay in the while loop to flush frames
          } else if (ret == AVERROR(EAGAIN)) {
            av_free_packet(&packet);
            continue;
          } else if (ret < 0) {
            LOG(ERROR) << "Error reading packet : " << ffmpegErrorStr(ret);
            return;
          }
          ipacket++;

          auto si = packet.stream_index;
          if (params.getAudio_ && audioStreamIndex_ >= 0 &&
              si == audioStreamIndex_) {
            // Audio packets can have multiple audio frames in a single packet
            while (packet.size > 0) {
              assert(audioCodecContext_ != nullptr);
              assert(convertCtx_ != nullptr);
              getAudioSample(
                  packet,
                  audioCodecContext_,
                  audioStreamFrame_,
                  convertCtx_,
                  callback,
                  params);
            }
          }

          if (si != videoStreamIndex_) {
            av_free_packet(&packet);
            continue;
          }
        }

        ret = avcodec_decode_video2(
            videoCodecContext_, videoStreamFrame_, &gotPicture, &packet);
        if (ret < 0) {
          LOG(ERROR) << "Error decoding video frame : " << ffmpegErrorStr(ret);
          return;
        }
        try {
          // Nothing to do without a picture
          if (!gotPicture) {
            av_free_packet(&packet);
            continue;
          }
          frameIndex++;

          long int frame_ts =
              av_frame_get_best_effort_timestamp(videoStreamFrame_);
          timestamp = frame_ts * av_q2d(videoStream_->time_base);
          if ((frame_ts >= start_ts && !mustDecodeAll) || mustDecodeAll) {
            /* process current frame if:
             * 1) We are not doing selective decoding and mustDecodeAll
             *    OR
             * 2) We are doing selective decoding and current frame
             *   timestamp is >= start_ts from where we start selective
             *   decoding*/
            // if reaching the next interval, update the current fps
            // and reset lastFrameTimestamp so the current frame could be
            // sampled (unless fps == SpecialFps::SAMPLE_NO_FRAME)
            if (itvlIter != params.intervals_.end() &&
                timestamp >= itvlIter->timestamp) {
              lastFrameTimestamp = -1.0;
              currFps = itvlIter->fps;
              prevTimestamp = itvlIter->timestamp;
              itvlIter++;
              if (itvlIter != params.intervals_.end() &&
                  prevTimestamp >= itvlIter->timestamp) {
                LOG(ERROR)
                    << "Sampling interval timestamps must be strictly ascending.";
                return;
              }
            }

            // keyFrame will bypass all checks on fps sampling settings
            bool keyFrame = params.keyFrames_ && videoStreamFrame_->key_frame;
            if (!keyFrame) {
              // if fps == SpecialFps::SAMPLE_NO_FRAME (0), don't sample at all
              if (currFps == SpecialFps::SAMPLE_NO_FRAME) {
                av_free_packet(&packet);
                continue;
              }

              // fps is considered reached in the following cases:
              // 1. lastFrameTimestamp < 0 - start of a new interval
              //    (or first frame)
              // 2. currFps == SpecialFps::SAMPLE_ALL_FRAMES (-1) - sample every
              //    frame
              // 3. timestamp - lastFrameTimestamp has reached target fps and
              //    currFps > 0 (not special fps setting)
              // different modes for fps:
              // SpecialFps::SAMPLE_NO_FRAMES (0):
              //     disable fps sampling, no frame sampled at all
              // SpecialFps::SAMPLE_ALL_FRAMES (-1):
              //     unlimited fps sampling, will sample at native video fps
              // SpecialFps::SAMPLE_TIMESTAMP_ONLY (-2):
              //     disable fps sampling, but will get the frame at specific
              //     timestamp
              // others (> 0): decoding at the specified fps
              bool fpsReached = lastFrameTimestamp < 0 ||
                  currFps == SpecialFps::SAMPLE_ALL_FRAMES ||
                  (currFps > 0 &&
                   timestamp >= lastFrameTimestamp + (1 / currFps));

              if (!fpsReached) {
                av_free_packet(&packet);
                continue;
              }
            }

            lastFrameTimestamp = timestamp;

            outputFrameIndex++;
            if (params.maximumOutputFrames_ != -1 &&
                outputFrameIndex >= params.maximumOutputFrames_) {
              // enough frames
              av_free_packet(&packet);
              break;
            }

            AVFrame* rgbFrame = av_frame_alloc();
            if (!rgbFrame) {
              LOG(ERROR) << "Error allocating AVframe";
              return;
            }

            try {
              // Determine required buffer size and allocate buffer
              int numBytes = avpicture_get_size(pixFormat, outWidth, outHeight);
              DecodedFrame::AvDataPtr buffer(
                  (uint8_t*)av_malloc(numBytes * sizeof(uint8_t)));

              int size = avpicture_fill(
                  (AVPicture*)rgbFrame,
                  buffer.get(),
                  pixFormat,
                  outWidth,
                  outHeight);

              sws_scale(
                  scaleContext_,
                  videoStreamFrame_->data,
                  videoStreamFrame_->linesize,
                  0,
                  videoCodecContext_->height,
                  rgbFrame->data,
                  rgbFrame->linesize);

              unique_ptr<DecodedFrame> frame = make_unique<DecodedFrame>();
              frame->width_ = outWidth;
              frame->height_ = outHeight;
              frame->data_ = std::move(buffer);
              frame->size_ = size;
              frame->index_ = frameIndex;
              frame->outputFrameIndex_ = outputFrameIndex;
              frame->timestamp_ = timestamp;
              frame->keyFrame_ = videoStreamFrame_->key_frame;

              callback.frameDecoded(std::move(frame));

              selectiveDecodedFrames++;
              av_frame_free(&rgbFrame);
            } catch (const std::exception&) {
              av_frame_free(&rgbFrame);
            }
          }
          av_frame_unref(videoStreamFrame_);
          av_frame_unref(audioStreamFrame_);
        } catch (const std::exception&) {
          av_frame_unref(videoStreamFrame_);
          av_frame_unref(audioStreamFrame_);
        }

        av_free_packet(&packet);
      } catch (const std::exception&) {
        av_free_packet(&packet);
      }
    } // of while loop
    callback.videoDecodingEnded(timestamp);

    // free all stuffs
    sws_freeContext(scaleContext_);
    swr_free(&convertCtx_);
    av_packet_unref(&packet);
    av_frame_free(&videoStreamFrame_);
    av_frame_free(&audioStreamFrame_);
    avcodec_close(videoCodecContext_);
    if (audioCodecContext_ != nullptr) {
      avcodec_close(audioCodecContext_);
    }
    avformat_close_input(&inputContext);
    avformat_free_context(inputContext);
  } catch (const std::exception&) {
    // In case of decoding error
    // free all stuffs
    sws_freeContext(scaleContext_);
    swr_free(&convertCtx_);
    av_packet_unref(&packet);
    av_frame_free(&videoStreamFrame_);
    av_frame_free(&audioStreamFrame_);
    avcodec_close(videoCodecContext_);
    avcodec_close(audioCodecContext_);
    avformat_close_input(&inputContext);
    avformat_free_context(inputContext);
  }
}

void VideoDecoder::decodeMemory(
    const string& videoName,
    const char* buffer,
    const int size,
    const Params& params,
    const int start_frm,
    Callback& callback) {
  VideoIOContext ioctx(buffer, size);
  decodeLoop(videoName, ioctx, params, start_frm, callback);
}

void VideoDecoder::decodeFile(
    const string& file,
    const Params& params,
    const int start_frm,
    Callback& callback) {
  VideoIOContext ioctx(file);
  decodeLoop(file, ioctx, params, start_frm, callback);
}

string VideoDecoder::ffmpegErrorStr(int result) {
  std::array<char, 128> buf;
  av_strerror(result, buf.data(), buf.size());
  return string(buf.data());
}

void FreeDecodedData(
    std::vector<std::unique_ptr<DecodedFrame>>& sampledFrames,
    std::vector<std::unique_ptr<DecodedAudio>>& sampledAudio) {
  // free the sampledFrames and sampledAudio
  for (int i = 0; i < sampledFrames.size(); i++) {
    DecodedFrame* p = sampledFrames[i].release();
    delete p;
  }
  for (int i = 0; i < sampledAudio.size(); i++) {
    DecodedAudio* p = sampledAudio[i].release();
    delete p;
  }
  sampledFrames.clear();
  sampledAudio.clear();
}

bool DecodeMultipleClipsFromVideo(
    const char* video_buffer,
    const std::string& video_filename,
    const int encoded_size,
    const Params& params,
    const int start_frm,
    const int clip_per_video,
    const std::vector<int>& clip_start_positions,
    const bool use_local_file,
    int& height,
    int& width,
    std::vector<unsigned char*>& buffer_rgb) {
  std::vector<std::unique_ptr<DecodedFrame>> sampledFrames;
  std::vector<std::unique_ptr<DecodedAudio>> sampledAudio;
  VideoDecoder decoder;

  CallbackImpl callback;
  // decoding from buffer or file
  if (!use_local_file) {
    decoder.decodeMemory(
        string("Memory Buffer"),
        video_buffer,
        encoded_size,
        params,
        start_frm,
        callback);
  } else {
    decoder.decodeFile(video_filename, params, start_frm, callback);
  }

  for (auto& frame : callback.frames) {
    sampledFrames.push_back(std::move(frame));
  }
  for (auto& audio_sample : callback.audio_samples) {
    sampledAudio.push_back(std::move(audio_sample));
  }

  for (int i = 0; i < buffer_rgb.size(); i++) {
    unsigned char* buff = buffer_rgb[i];
    delete[] buff;
  }
  buffer_rgb.clear();

  if (sampledFrames.size() < params.num_of_required_frame_) {
    LOG(ERROR)
        << "The video seems faulty and we could not decode enough frames: "
        << sampledFrames.size() << " VS " << params.num_of_required_frame_;
    FreeDecodedData(sampledFrames, sampledAudio);
    return true;
  }
  if (sampledFrames.size() == 0) {
    LOG(ERROR) << "The samples frames have size 0, no frame to process";
    FreeDecodedData(sampledFrames, sampledAudio);
    return true;
  }
  height = sampledFrames[0]->height_;
  width = sampledFrames[0]->width_;
  float sample_stepsz = (clip_per_video <= 1)
      ? 0
      : (float(sampledFrames.size() - params.num_of_required_frame_) /
         (clip_per_video - 1));

  int image_size = 3 * height * width;
  int clip_size = params.num_of_required_frame_ * image_size;
  // get the RGB frames for each clip
  if (clip_start_positions.size() > 0) {
    for (int i = 0; i < clip_start_positions.size(); i++) {
      unsigned char* buffer_rgb_ptr = new unsigned char[clip_size];
      int clip_start = clip_start_positions[i];
      for (int j = 0; j < params.num_of_required_frame_; j++) {
        memcpy(
            buffer_rgb_ptr + j * image_size,
            (unsigned char*)sampledFrames[j + clip_start]->data_.get(),
            image_size * sizeof(unsigned char));
      }
      buffer_rgb.push_back(buffer_rgb_ptr);
    }
  } else {
    for (int i = 0; i < clip_per_video; i++) {
      unsigned char* buffer_rgb_ptr = new unsigned char[clip_size];
      int clip_start = floor(i * sample_stepsz);
      for (int j = 0; j < params.num_of_required_frame_; j++) {
        memcpy(
            buffer_rgb_ptr + j * image_size,
            (unsigned char*)sampledFrames[j + clip_start]->data_.get(),
            image_size * sizeof(unsigned char));
      }
      buffer_rgb.push_back(buffer_rgb_ptr);
    }
  }
  FreeDecodedData(sampledFrames, sampledAudio);

  return true;
}

} // namespace caffe2
