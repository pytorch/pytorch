.. role:: hidden
    :class: hidden-section

torch.nn.functional
===================

.. currentmodule:: torch.nn.functional

Convolution functions
----------------------------------

:hidden:`conv1d`
~~~~~~~~~~~~~~~~

.. autofunction:: conv1d

:hidden:`conv2d`
~~~~~~~~~~~~~~~~

.. autofunction:: conv2d

:hidden:`conv3d`
~~~~~~~~~~~~~~~~

.. autofunction:: conv3d

:hidden:`conv_transpose1d`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: conv_transpose1d

:hidden:`conv_transpose2d`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: conv_transpose2d

:hidden:`conv_transpose3d`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: conv_transpose3d

:hidden:`unfold`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: unfold

:hidden:`fold`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: fold

Pooling functions
----------------------------------

:hidden:`avg_pool1d`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: avg_pool1d

:hidden:`avg_pool2d`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: avg_pool2d

:hidden:`avg_pool3d`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: avg_pool3d

:hidden:`max_pool1d`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: max_pool1d

:hidden:`max_pool2d`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: max_pool2d

:hidden:`max_pool3d`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: max_pool3d

:hidden:`max_unpool1d`
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: max_unpool1d

:hidden:`max_unpool2d`
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: max_unpool2d

:hidden:`max_unpool3d`
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: max_unpool3d

:hidden:`lp_pool1d`
~~~~~~~~~~~~~~~~~~~

.. autofunction:: lp_pool1d

:hidden:`lp_pool2d`
~~~~~~~~~~~~~~~~~~~

.. autofunction:: lp_pool2d

:hidden:`adaptive_max_pool1d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adaptive_max_pool1d

:hidden:`adaptive_max_pool2d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adaptive_max_pool2d

:hidden:`adaptive_max_pool3d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adaptive_max_pool3d

:hidden:`adaptive_avg_pool1d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adaptive_avg_pool1d

:hidden:`adaptive_avg_pool2d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adaptive_avg_pool2d

:hidden:`adaptive_avg_pool3d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: adaptive_avg_pool3d


Non-linear activation functions
-------------------------------

:hidden:`threshold`
~~~~~~~~~~~~~~~~~~~

.. autofunction:: threshold
.. autofunction:: threshold_


:hidden:`relu`
~~~~~~~~~~~~~~

.. autofunction:: relu
.. autofunction:: relu_

:hidden:`hardtanh`
~~~~~~~~~~~~~~~~~~

.. autofunction:: hardtanh
.. autofunction:: hardtanh_

:hidden:`relu6`
~~~~~~~~~~~~~~~

.. autofunction:: relu6

:hidden:`elu`
~~~~~~~~~~~~~

.. autofunction:: elu
.. autofunction:: elu_

:hidden:`selu`
~~~~~~~~~~~~~~

.. autofunction:: selu

:hidden:`celu`
~~~~~~~~~~~~~~

.. autofunction:: celu

:hidden:`leaky_relu`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: leaky_relu
.. autofunction:: leaky_relu_

:hidden:`prelu`
~~~~~~~~~~~~~~~

.. autofunction:: prelu

:hidden:`rrelu`
~~~~~~~~~~~~~~~

.. autofunction:: rrelu
.. autofunction:: rrelu_

:hidden:`glu`
~~~~~~~~~~~~~~~

.. autofunction:: glu

:hidden:`gelu`
~~~~~~~~~~~~~~~

.. autofunction:: gelu

:hidden:`logsigmoid`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: logsigmoid

:hidden:`hardshrink`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hardshrink

:hidden:`tanhshrink`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: tanhshrink

:hidden:`softsign`
~~~~~~~~~~~~~~~~~~

.. autofunction:: softsign

:hidden:`softplus`
~~~~~~~~~~~~~~~~~~

.. autofunction:: softplus

:hidden:`softmin`
~~~~~~~~~~~~~~~~~

.. autofunction:: softmin

:hidden:`softmax`
~~~~~~~~~~~~~~~~~

.. autofunction:: softmax

:hidden:`softshrink`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: softshrink

:hidden:`gumbel_softmax`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gumbel_softmax

:hidden:`log_softmax`
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: log_softmax

:hidden:`tanh`
~~~~~~~~~~~~~~

.. autofunction:: tanh

:hidden:`sigmoid`
~~~~~~~~~~~~~~~~~

.. autofunction:: sigmoid

Normalization functions
-----------------------

:hidden:`batch_norm`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: batch_norm

:hidden:`instance_norm`
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: instance_norm

:hidden:`layer_norm`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: layer_norm

:hidden:`local_response_norm`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: local_response_norm

:hidden:`normalize`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: normalize

Linear functions
----------------

:hidden:`linear`
~~~~~~~~~~~~~~~~

.. autofunction:: linear

:hidden:`bilinear`
~~~~~~~~~~~~~~~~~~

.. autofunction:: bilinear

Dropout functions
-----------------

:hidden:`dropout`
~~~~~~~~~~~~~~~~~

.. autofunction:: dropout

:hidden:`alpha_dropout`
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: alpha_dropout

:hidden:`dropout2d`
~~~~~~~~~~~~~~~~~~~

.. autofunction:: dropout2d

:hidden:`dropout3d`
~~~~~~~~~~~~~~~~~~~

.. autofunction:: dropout3d

Sparse functions
----------------------------------

:hidden:`embedding`
~~~~~~~~~~~~~~~~~~~

.. autofunction:: embedding

:hidden:`embedding_bag`
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: embedding_bag

:hidden:`one_hot`
~~~~~~~~~~~~~~~~~

.. autofunction:: one_hot

Distance functions
----------------------------------

:hidden:`pairwise_distance`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pairwise_distance

:hidden:`cosine_similarity`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosine_similarity

:hidden:`pdist`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pdist


Loss functions
--------------

:hidden:`binary_cross_entropy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: binary_cross_entropy

:hidden:`binary_cross_entropy_with_logits`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: binary_cross_entropy_with_logits

:hidden:`poisson_nll_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: poisson_nll_loss

:hidden:`cosine_embedding_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosine_embedding_loss

:hidden:`cross_entropy`
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cross_entropy

:hidden:`ctc_loss`
~~~~~~~~~~~~~~~~~~

.. autofunction:: ctc_loss

:hidden:`hinge_embedding_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hinge_embedding_loss

:hidden:`kl_div`
~~~~~~~~~~~~~~~~

.. autofunction:: kl_div

:hidden:`l1_loss`
~~~~~~~~~~~~~~~~~

.. autofunction:: l1_loss

:hidden:`mse_loss`
~~~~~~~~~~~~~~~~~~

.. autofunction:: mse_loss

:hidden:`margin_ranking_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: margin_ranking_loss

:hidden:`multilabel_margin_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: multilabel_margin_loss

:hidden:`multilabel_soft_margin_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: multilabel_soft_margin_loss

:hidden:`multi_margin_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: multi_margin_loss

:hidden:`nll_loss`
~~~~~~~~~~~~~~~~~~

.. autofunction:: nll_loss

:hidden:`smooth_l1_loss`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: smooth_l1_loss

:hidden:`soft_margin_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: soft_margin_loss

:hidden:`triplet_margin_loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: triplet_margin_loss

Vision functions
----------------

:hidden:`pixel_shuffle`
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pixel_shuffle

:hidden:`pad`
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pad

:hidden:`interpolate`
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: interpolate

:hidden:`upsample`
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: upsample

:hidden:`upsample_nearest`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: upsample_nearest

:hidden:`upsample_bilinear`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: upsample_bilinear

:hidden:`grid_sample`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: grid_sample

:hidden:`affine_grid`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: affine_grid

DataParallel functions (multi-GPU, distributed)
-----------------------------------------------

:hidden:`data_parallel`
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: torch.nn.parallel.data_parallel


