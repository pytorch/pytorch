#ifndef CAFFE2_OPERATORS_CROP_AND_RESIZE_OP_H_
#define CAFFE2_OPERATORS_CROP_AND_RESIZE_OP_H_

#include "caffe2/core/operator.h"

namespace caffe2 {
    
enum CropAndResizeMethod
{
    BILINEAR = 0,
    NEAREST = 1
};

template<typename T, class Context>
class CropAndResizeOp final : public Operator<Context> {
    
 public:
    
    USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit CropAndResizeOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        crop_width_(
            this->template GetSingleArgument<int>("crop_width", 1)),
        crop_height_(
            this->template GetSingleArgument<int>("crop_height", 1)),
        extrapolation_value_(
            this->template GetSingleArgument<float>("extrapolation_value", .0f)),
        method_(
            static_cast<CropAndResizeMethod>(
            this->template GetSingleArgument<int>("method", 
                static_cast<int>(CropAndResizeMethod::BILINEAR))))
    {
        CAFFE_ENFORCE_GT(crop_width_, 0);
        CAFFE_ENFORCE_GT(crop_height_, 0);
        
        CAFFE_ENFORCE(method_ == CropAndResizeMethod::BILINEAR || method_ == CropAndResizeMethod::NEAREST);
    }
    
    virtual ~CropAndResizeOp() {}
    
    virtual bool RunOnDevice() override;
        
 private:
 
    int crop_width_;
    int crop_height_;
 
    float extrapolation_value_;
    
    CropAndResizeMethod method_;
    
    INPUT_TAGS(FM, BOXES);
 
};

template<typename T, class Context>
class CropAndResizeGradientOp final : public Operator<Context> {

public:

    USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit CropAndResizeGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        crop_width_(
            this->template GetSingleArgument<int>("crop_width", 1)),
        crop_height_(
            this->template GetSingleArgument<int>("crop_height", 1)),
        extrapolation_value_(
            this->template GetSingleArgument<float>("extrapolation_value", .0f)),
        method_(
            static_cast<CropAndResizeMethod>(
            this->template GetSingleArgument<int>("method", 
                    static_cast<int>(CropAndResizeMethod::BILINEAR))))
    {
        CAFFE_ENFORCE_GT(crop_width_, 0);
        CAFFE_ENFORCE_GT(crop_height_, 0);

        CAFFE_ENFORCE(method_ == CropAndResizeMethod::BILINEAR || method_ == CropAndResizeMethod::NEAREST);
    }

    virtual ~CropAndResizeGradientOp() {}

    virtual bool RunOnDevice() override;

private:

    int crop_width_;
    int crop_height_;

    float extrapolation_value_;

    CropAndResizeMethod method_;

    INPUT_TAGS(FM, BOXES);

};

 
} // namespace caffe2

#endif // CAFFE2_OPERATORS_CROP_AND_RESIZE_OP_H_

