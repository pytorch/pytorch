
#pragma once

#include "GLTexture.h"
#include "caffe2/core/logging.h"

#include <functional>
#include <vector>

template <typename T>
class GLImage {
 public:
  const int width;
  const int height;
  const int channels;
  const int data_size;

  const int tile_x;
  const int tile_y;
  const int texture_width;
  const int texture_height;
  const int slices;

  const std::vector<const GLTexture*> textures;

  constexpr static int slice_channels = 4;

  static constexpr int channels_to_slices(int channels, int tile_x, int tile_y) {
    return ((channels + slice_channels - 1) / slice_channels + tile_x * tile_y - 1) /
           (tile_x * tile_y);
  }

  static const std::vector<const GLTexture*> allocate_textures(
      int slices, std::function<const GLTexture*(int slice)> texture_loader) {
    std::vector<const GLTexture*> textures;
    for (int i = 0; i < slices; i++) {
      textures.push_back(texture_loader(i));
    }
    return textures;
  }

  GLImage(int _width,
          int _height,
          int _channels,
          int _tile_x,
          int _tile_y,
          std::function<const GLTexture*(int slice)> texture_loader)
      : width(_width),
        height(_height),
        channels(_channels),
        data_size(sizeof(T)),
        tile_x(_tile_x),
        tile_y(_tile_y),
        texture_width(_width * _tile_x),
        texture_height(_height * _tile_y),
        slices(channels_to_slices(_channels, _tile_x, _tile_y)),
        textures(allocate_textures(slices, texture_loader)) {
    CAFFE_ENFORCE_EQ(
        slices, ((channels + 3) / 4 + tile_x * tile_y - 1) / (tile_x * tile_y));
  }

  GLImage(int _width,
          int _height,
          int _channels,
          int _tile_x,
          int _tile_y,
          bool _destroy,
          std::function<const GLTexture*(int slice)> texture_loader)
      : width(_width),
        height(_height),
        channels(_channels),
        data_size(sizeof(T)),
        tile_x(_tile_x),
        tile_y(_tile_y),
        texture_width(_width * _tile_x),
        texture_height(_height * _tile_y),
        slices(channels_to_slices(_channels, _tile_x, _tile_y)),
        textures(allocate_textures(slices, texture_loader)) {
    CAFFE_ENFORCE_EQ(slices * tile_x * tile_y, (channels + 3) / 4);
  }

  GLImage()
      : width(0),
        height(0),
        channels(0),
        data_size(sizeof(T)),
        tile_x(0),
        tile_y(0),
        texture_width(0),
        texture_height(0),
        slices(0){};

  virtual ~GLImage() {
    gl_log(GL_VERBOSE, "deleting GLImage\n");
    for (auto&& texture : textures) {
      delete texture;
    }
  }
};

template <typename T>
class GLImageVector {
 private:
  std::vector<GLImage<T>*> images_;
  int num_images_ = 0;
  int width_ = 0;
  int height_ = 0;
  int channels_ = 0;
  int tile_x_ = 0;
  int tile_y_ = 0;

 public:
  GLImage<T>* operator[](int index) const {
    CAFFE_ENFORCE_LT(index, num_images_, "Out of bounds when accessing GLImageVector");
    return images_[index];
  }

  void push_back(GLImage<T>* image) {
    CAFFE_ENFORCE_EQ(image->channels, channels_);
    CAFFE_ENFORCE_EQ(image->width, width_);
    CAFFE_ENFORCE_EQ(image->height, height_);
    CAFFE_ENFORCE_EQ(image->tile_x, tile_x_);
    CAFFE_ENFORCE_EQ(image->tile_y, tile_y_);
    images_.push_back(image);
    CAFFE_ENFORCE_LE(images_.size(), num_images_);
  }

  int size() const { return images_.size(); }
  int channels() const { return channels_; }
  int width() const { return width_; }
  int height() const { return height_; }
  int tile_x() const { return tile_x_; }
  int tile_y() const { return tile_y_; }
  int slices() const { return size() > 0 ? images_[0]->slices : 0; }

  GLImageVector(int num_images, int width, int height, int channels, int tile_x = 1, int tile_y = 1)
      : num_images_(num_images),
        width_(width),
        height_(height),
        channels_(channels),
        tile_x_(tile_x),
        tile_y_(tile_y) {}

  GLImageVector() {}

  ~GLImageVector() {
    for (int i = 0; i < images_.size(); i++) {
      delete images_[i];
    }
  }
};
