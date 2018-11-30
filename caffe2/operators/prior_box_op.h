#ifndef PRIOR_BOX_OP_H_
#define PRIOR_BOX_OP_H_

#include "caffe2/core/operator.h"

namespace caffe2 {

    template<typename T, typename Context>
    class PriorBoxOp final : public Operator<Context> {

    public:

        USE_OPERATOR_CONTEXT_FUNCTIONS;

        PriorBoxOp(const OperatorDef& operator_def, Workspace* ws)
            : Operator<Context>(operator_def, ws),
            widths_(
                this->template GetRepeatedArgument<float>("widths")),
            heights_(
                this->template GetRepeatedArgument<float>("heights")),
            step_w_(
                this->template GetSingleArgument<float>("step_w", .0f)),
            step_h_(
                this->template GetSingleArgument<float>("step_h", .0f)),
            step_(
                this->template GetSingleArgument<float>("step", .0f)),
            offset_(
                this->template GetSingleArgument<float>("offset", .5f)),
            clip_(
                this->template GetSingleArgument<bool>("clip", false))
        {

            CAFFE_ENFORCE_GT(widths_.size(), 0);
            CAFFE_ENFORCE_EQ(widths_.size(), heights_.size());

            num_priors_ = widths_.size();

            CAFFE_ENFORCE_GE(step_h_, .0f);
            CAFFE_ENFORCE_GE(step_w_, .0f);
            CAFFE_ENFORCE_GE(step_, .0f);

            if (step_h_ > 0 || step_w_ > 0) {
                CAFFE_ENFORCE(step_h_ > 0 && step_w_ > 0);
                CAFFE_ENFORCE(std::fabs(step_) < std::numeric_limits<float>::epsilon());
            }
            else if (step_ > 0) {
                step_h_ = step_;
                step_w_ = step_;
            }

            CAFFE_ENFORCE_GE(offset_, .0f);

        }

        virtual ~PriorBoxOp() {}

        virtual bool RunOnDevice() override;

    private:

        std::vector<float> widths_;
        std::vector<float> heights_;

        float step_w_;
        float step_h_;
        float step_;
        float offset_;

        bool clip_;

        int num_priors_;

        INPUT_TAGS(FM, IMAGE);
    };

} // namespace caffe2

#endif // PRIOR_BOX_OP_H_

