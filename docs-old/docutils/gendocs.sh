SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd $SCRIPT_DIR

# module
#python doc2md.py torch.nn Module --title Module --no-toc >../nn_module.md

# containers
echo "## Containers" > ../nn_container.md
python doc2md.py torch.nn Container --title Container --no-toc    >>../nn_container.md
python doc2md.py torch.nn Sequential --title Sequential --no-toc >>../nn_container.md

# convolution
echo "## Convolution Layers" > ../nn_convolution.md
echo Conv1d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc          >>../nn_convolution.md
echo Conv2d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc          >>../nn_convolution.md
echo ConvTranspose2d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc >>../nn_convolution.md
echo Conv3d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc          >>../nn_convolution.md
echo ConvTranspose3d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc >>../nn_convolution.md

# pooling
echo "## Pooling Layers" > ../nn_pooling.md
echo MaxPool1d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc           >>../nn_pooling.md
echo MaxPool2d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc           >>../nn_pooling.md
echo MaxPool3d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc           >>../nn_pooling.md
echo MaxUnpool2d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc         >>../nn_pooling.md
echo MaxUnpool3d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc         >>../nn_pooling.md
echo AvgPool2d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc           >>../nn_pooling.md
echo AvgPool3d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc           >>../nn_pooling.md
echo FractionalMaxPool2d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc >>../nn_pooling.md
echo LPPool2d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc            >>../nn_pooling.md

# activations
echo "## Non-linearities" > ../nn_activation.md
echo ReLU | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc            >>../nn_activation.md
echo ReLU6 | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc           >>../nn_activation.md
echo Threshold | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc       >>../nn_activation.md
echo Hardtanh | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc        >>../nn_activation.md
echo Sigmoid | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc         >>../nn_activation.md
echo Tanh | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc            >>../nn_activation.md
echo ELU | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc             >>../nn_activation.md
echo LeakyReLU | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc       >>../nn_activation.md
echo LogSigmoid | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc      >>../nn_activation.md
echo Softplus | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc        >>../nn_activation.md
echo Softshrink | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc      >>../nn_activation.md
echo PReLU | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc           >>../nn_activation.md
echo Softsign | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc        >>../nn_activation.md
echo Tanhshrink | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc      >>../nn_activation.md
echo Softmin | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc         >>../nn_activation.md
echo Softmax | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc         >>../nn_activation.md
echo Softmax2d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc       >>../nn_activation.md
echo LogSoftmax | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc      >>../nn_activation.md

# normalization
echo "## Normalization layers" > ../nn_normalization.md
echo BatchNorm1d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc >>../nn_normalization.md
echo BatchNorm2d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc >>../nn_normalization.md
echo BatchNorm3d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc >>../nn_normalization.md

# recurrentnet
echo "## Recurrent layers" > ../nn_recurrent.md
echo RNN | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc             >>../nn_recurrent.md
echo LSTM | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc            >>../nn_recurrent.md
echo GRU | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc             >>../nn_recurrent.md
echo RNNCell | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc         >>../nn_recurrent.md
echo LSTMCell | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc        >>../nn_recurrent.md
echo GRUCell | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc         >>../nn_recurrent.md

# linear
echo "## Linear layers" > ../nn_linear.md
echo Linear | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc      >>../nn_linear.md

# dropout
echo "## Dropout layers" > ../nn_dropout.md
echo Dropout | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc   >>../nn_dropout.md
echo Dropout2d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc >>../nn_dropout.md
echo Dropout3d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc >>../nn_dropout.md

# Sparse
echo "## Sparse layers" > ../nn_sparse.md
echo Embedding | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc              >>../nn_sparse.md

# loss_functions
echo "## Loss functions" > ../nn_loss.md
echo L1Loss | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc                        >>../nn_loss.md
echo MSELoss | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc                       >>../nn_loss.md
echo CrossEntropyLoss | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc              >>../nn_loss.md
echo NLLLoss | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc                       >>../nn_loss.md
echo NLLLoss2d | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc                     >>../nn_loss.md
echo KLDivLoss | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc                     >>../nn_loss.md
echo BCELoss | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc                       >>../nn_loss.md
echo MarginRankingLoss | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc             >>../nn_loss.md
echo HingeEmbeddingLoss | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc            >>../nn_loss.md
echo MultiLabelMarginLoss | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc          >>../nn_loss.md
echo SmoothL1Loss | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc                  >>../nn_loss.md
echo SoftMarginLoss | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc                >>../nn_loss.md
echo MultiLabelSoftMarginLoss | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc      >>../nn_loss.md
echo CosineEmbeddingLoss | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc           >>../nn_loss.md
echo MultiMarginLoss | xargs -I {} python doc2md.py torch.nn {} --title {} --no-toc               >>../nn_loss.md

popd
