.. _gradcheck-mechanics:

Gradcheck mechanics
===================

This note presents an overview of how the :meth:`~torch.autograd.gradcheck` and :meth:`~torch.autograd.gradgradcheck` functions work.

It will cover both forward and backward mode AD as well as complex valued functions and higher order derivatives.

.. contents:: :local:
    :depth: 2

Notations and background informations
-------------------------------------

Throughout this note, we will use :math:`x`, :math:`y`, :math:`a`, :math:`b`, :math:`v`, :math:`u`, :math:`ur` and :math:`ui` as real valued vectors and :math:`z` is a complex valued vector.
:math:`N` and :math:`M` are two integers that we will use for the dimension of the input and output space respectively.

We will use :math:`f: \mathcal{R}^N \to \mathcal{R}^M` as our basic real to real function such that :math:`y = f(x)`.
We will use :math:`g: \mathcal{C}^N \to \mathcal{R}^M` as our basic complex to real function such that :math:`y = g(z)`.
We will also use :math:`z = a + i b`.


For the simple real to real case, we write as :math:`J_f` the jacobian matrix associated with :math:`f` of size :math:`M \times N`.
This matrix contains all the partial derivatives such that the entry at position :math:`(i, j)` contains :math:`\frac{\partial y_i}{\partial x_j}`.
Backward mode AD is then computing, for a given vector :math:`v` of size :math:`M`, the quantity :math:`v^T J_f`.
Forward mode AD on the other hand is computing, for a given vector :math:`u` of size :math:`N`, the quantity :math:`J_f u`.

For functions that contain complex values, the story is a lot more complex. We only provide the gist here and the full description can be found at :ref:`complex_autograd-doc`.

Proper complex derivatives are too restrictive for most functions we care about. So we turn to Wirtinger calculus.
In a basic setting of Wirtinger calculus, the chain rule required access to both the Wirtinger derivative (called :math:`W` below) and the Conjugate Wirtinger derivative (called :math:`CW` below). This means that in the general case of Wirtinger calculus, the values that we would need to "backward through the graph" in backward mode AD would need to be twice the size of the "forward value".

To avoid this problem, for backward mode AD, we always work under the assumption that the current function is part of a bigger function whose output is in :math:`\mathcal{R}`. Note that this assumption means that all the intermediary gradients we compute during the backward pass are also corresponding to functions whose output is in :math:`\mathcal{R}`.
Under this assumption, we can show that :math:`W = CW^*` (we use :math:`*` to denote complex conjugation here) and so only one of the two values actually need to be "backwarded through the graph" as the other one can easily be recovered.
To simplify internal computations, PyTorch uses :math:`2 * CW` as the value it backwards and returns when the user asks for gradients.
Similarly to the real case, when the output is actually in :math:`\mathcal{R}^M`, the whole matrix cannot be recovered in one backward pass but only :math:`v^T (2 * CW)` for a given vector :math:`v \in \mathcal{R}^M`.

For forward mode AD, we use a similar logic, in this case, assuming that the function is part of a larger function whose input is in :math:`\mathcal{R}`. Under this assumption, we can make similar claim that every intermediary result corresponds to a function whose input is in :math:`\mathcal{R}` and in this case, we can show that :math:`W = CW` for the intermediary functions.
To make sure the forward and backward mode compute the same quantities, the forward mode also computes :math:`2 * CW`.
Similarly to the real case, when the input is actually in :math:`\mathcal{R}^N`, the whole matrix cannot be recovered in one backward pass but only :math:`(2 * CW) u` for a given vector :math:`u \in \mathcal{R}^N`.


Current backward mode gradcheck behavior
----------------------------------------

Functions in :math:`\mathcal{R}^N \to \mathcal{R}^M`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We want to test a function :math:`f: \mathcal{R}^N \to \mathcal{R}^M, x \to y`.
To do so, we reconstruct the full jacobian matrix :math:`J_f` of size :math:`M \times N` in two ways.
The analytical version uses our backward mode AD while the numerical version uses finite difference.
We then compare each element of the two reconstructed jacobian matrices to ensure they do match.

Slow real input numerical evaluation
""""""""""""""""""""""""""""""""""""

If we consider the most basic case of a one dimensional function (:math:`N = M = 1`), the we can use the basic finite difference formula from `the wikipedia article <https://en.wikipedia.org/wiki/Finite_difference>`_ (we use the central difference for better numerical properties):

.. math::
    \frac{\partial y}{\partial x} \approx \frac{f(x + eps) - f(x - eps)}{2 * eps}

This formula easily generalize for multiple outputs (:math:`M \gt 1`) by having :math:`\frac{\partial y}{\partial x}` be a column vector of size :math:`M \times 1` like :math:`f(x + eps)`. In that case, the above formula can be re-used as-is and approximate the full Jacobian matrix with only two evaluations of the user function.

It is more expensive when considering the case with multiple inputs (:math:`N \gt 1`). To handle this case, we actually loop over all the inputs one after the other and apply the :math:`eps` perturbation for each element of :math:`x` one after the other. This allows us to reconstruct the :math:`J_f` matrix column by column.

Slow real input analytical evaluation
"""""""""""""""""""""""""""""""""""""

In this case, we use the fact, as described above that backward mode AD allows us to compute :math:`v^T J_f`.
For functions that have a single output, we simply use :math:`v = 1` to recover the full Jacobian matrix with a single backward pass.

In the general case of multiple output, we again resort to a for-loop but this one iterates over the outputs where each :math:`v` is a one-hot vector corresponding to each output one after the other. This allows to reconstruct the :math:`J_f` matrix row by row.

Functions in :math:`\mathcal{C}^N \to \mathcal{R}^M`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We want to test a function :math:`g: \mathcal{C}^N \to \mathcal{R}^M, z \to y` with :math:`z = a + i b`.
In this case, we reconstruct the (complex valued) matrix that contains :math:`2 * CW`.

Slow complex input numerical evaluation
"""""""""""""""""""""""""""""""""""""""

When considering the simple case where :math:`N = M = 1`, we know (from first page of chapter 3 of `this research paper <https://arxiv.org/pdf/1701.00392.pdf>`_) that:

.. math::
    CW := \frac{\partial y}{\partial z^*} = \frac{1}{2} * (\frac{\partial y}{\partial a} + i \frac{\partial y}{\partial b})

It is important to note that in this formula, :math:`\frac{\partial y}{\partial a}` and :math:`\frac{\partial y}{\partial b}` are simple :math:`\mathcal{R} \to \mathcal{R}` derivatives.
To evaluate this numerically, we thus approximate :math:`\frac{\partial y}{\partial a}` and :math:`\frac{\partial y}{\partial b}` using the method described above for the real to real case, compute the :math:`CW` matrix and then multiply it by :math:`2`.

Note that the current code computes this value in a slightly convoluted way:

.. code:: python

    # Code from https://github.com/pytorch/pytorch/blob/58eb23378f2a376565a66ac32c93a316c45b6131/torch/autograd/gradcheck.py#L99-L105
    # Notation changes in this code block:
    # s here is y above
    # x, y here are a, b above

    ds_dx = compute_gradient(eps)
    ds_dy = compute_gradient(eps * 1j)
    # conjugate wirtinger derivative
    conj_w_d = 0.5 * (ds_dx + ds_dy * 1j)
    # wirtinger derivative
    w_d = 0.5 * (ds_dx - ds_dy * 1j)
    d[d_idx] = grad_out.conjugate() * conj_w_d + grad_out * w_d.conj()

    # Since grad_out is always 1, and W and CW are complex conjugate of each other, the last line ends up computing exactly `conj_w_d + w_d.conj() = conj_w_d + conj_w_d = 2 * conj_w_d`.


Slow complex input analytical evaluation
""""""""""""""""""""""""""""""""""""""""

For this case, since the backward mode AD is computing exactly twice the :math:`CW` derivative already, we simply use the same trick as for the real to real case and reconstruct the matrix row by row when there are multiple real outputs.

Functions with outputs in :math:`\mathcal{C}^M`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this case, the assumption we have about the computation of the :math:`CW` does not hold (output is not real) and so we cannot compute it directly to compare it.
To solve this, we will replace the test of the function :math:`h: \mathcal{P}^N \to \mathcal{C}^M` (where :math:`\mathcal{P}` can be either :math:`\mathcal{R}` or :math:`\mathcal{C}`), with two functions :math:`hr` and :math:`hi` such that: :math:`hr(q) = real(f(q))` and :math:`hi(q) = imag(f(q))` where :math:`q \in \mathcal{P}`.
We then do a basic gradcheck for both :math:`hr` and :math:`hi` using either the real to real or complex to real case described above, depending on :math:`\mathcal{P}`.

Note that in the current code does not create these functions explicitly but perform the chain rule with the :math:`real` or :math:`imag` functions manually by passing the :math:`\text{grad\_out}` arguments to the different functions.
When :math:`\text{grad\_out} = 1`, then we are considering :math:`hr`.
When :math:`\text{grad\_out} = i`, then we are considering :math:`hi`.


Fast backward mode gradcheck
----------------------------

While the above formulation of gradcheck is great both to ensure correctness and debug-ability, it is very slow the reconstruct full Jacobian matrices.
This section presents a way to perform gradcheck in a faster way without effecting its correctness.
The debug-ability can be recovered easily by adding special code when we detect an error in the computed gradients.

The high level strategy here is going to find a scalar quantity that can be computed efficiently by both the numerical and analytical methods and that represent well enough the full matrix computed by the slow code to ensure that it will catch any discrepancy in the Jacobians.

Fast gradcheck for functions in :math:`\mathcal{R}^N \to \mathcal{R}^M`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this case, the scalar quantity that we want to compute is :math:`v^T J_f u` for a given random vector :math:`v \in \mathcal{R}^M` and a random unit norm vector :math:`u \in \mathcal{R}^N`.

For the numerical evaluation, we can efficiently compute :math:`J_f u \approx \frac{f(x + u * eps) - f(x - u * eps)}{2 * eps}`. We then perform the dot product between this vector and :math:`v` to get the scalar value of interest.

For the analytical version, we can use backward mode AD to compute :math:`v^T J_f` directly. We then perform the dot product with :math:`u` to get the expected value.

Fast gradcheck for functions in :math:`\mathcal{C}^N \to \mathcal{R}^M`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a similar way to the real to real case, we want to perform a reduction of the full matrix. But the :math:`2 * CW` matrix is complex valued and so in this case, we will compare to complex scalars.

Due to some constraints on what we can compute efficiently in the numerical case and to keep the number of numerical evaluations to a minimum, we compute the following (albeit surprising) scalar value:

.. math::
    s := 2 * v^T (real(CW) ur + i * imag(CW) ui)

where :math:`v \in \mathcal{R}^M`, :math:`ur \in \mathcal{R}^N` and :math:`ui \in \mathcal{R}^N`.

Fast complex input numerical evaluation
"""""""""""""""""""""""""""""""""""""""

We first consider how to compute :math:`s` with a numerical method. To do so, keeping in mind that we're considering :math:`g: \mathcal{C}^N \to \mathcal{R}^M, z \to y` with :math:`z = a + i b`, and that :math:`CW = \frac{1}{2} * (\frac{\partial y}{\partial a} + i \frac{\partial y}{\partial b})`,  we rewrite it as follows:

.. math::
    \begin{aligned}
        s &= 2 * v^T (real(CW) ur + i * imag(CW) ui)
          &= 2 * v^T (\frac{1}{2} * \frac{\partial y}{\partial a} ur + i * \frac{1}{2} * \frac{\partial y}{\partial b} ui)
          &= v^T (\frac{\partial y}{\partial a} ur + i * \frac{\partial y}{\partial b} ui)
          &= v^T ((\frac{\partial y}{\partial a} ur) + i * (\frac{\partial y}{\partial b} ui))
    \end{aligned}

In this formula, we can see that :math:`\frac{\partial y}{\partial a} ur` and :math:`\frac{\partial y}{\partial b} ui` can be evaluated the same way as the fast version for the real to real case.
Once these real-valued quantities have been computed, we can reconstruct the complex vector on the right side and do a dot product with the real valued :math:`v` vector.

Fast complex input analytical evaluation
""""""""""""""""""""""""""""""""""""""""

For the analytical case, things are simpler and we rewrite the formula as:

.. math::
    \begin{aligned}
        s &= 2 * v^T (real(CW) ur + i * imag(CW) ui)
          &= v^T real(2 * CW) ur + i * v^T imag(2 * CW) ui)
          &= real(v^T (2 * CW)) ur + i * imag(v^T (2 * CW)) ui
    \end{aligned}

We can thus use the fact that the backward mode AD provides us with an efficient way to compute :math:`v^T (2 * CW)` and then perform a dot product of the real part with :math:`ur` and the imaginary part with :math:`ui` before reconstructing the final complex scalar :math:`s`.

Why not use a complex :math:`u`
"""""""""""""""""""""""""""""""

At this point, you might be wondering why we did not select a complex :math:`u` and just performed the reduction :math:`2 * v^T CW u'`.
The problem is that when doing the numerical evaluation, considering :math:`u' = ur' + i ui'`, we would need to compute:

.. math::
    \begin{aligned}
        2*CW u' &= (dy/da + i dy/db)(ur' + i ui')
                &= dy/da ur' + i dy/da ui' + i dy/db ur' - dy/db ui'
    \end{aligned}

Which would require four evaluations of real to real finite difference (twice as much compared to the approached proposed above).
Since this approach does not provide any correctness benefit, we use the formulation above.


Fast gradcheck for functions with outputs in :math:`\mathcal{C}^M`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Just like in the slow case, we consider two real valued function and use the appropriate rule from above for each function.

Gradgrad check implementation
-----------------------------

PyTorch also provide a utility to verify second order gradients. The goal here is to make sure that the backward implementation is also properly differentiable and computes the right thing.

This feature is implemented by considering the function :math:`F: x, v \to v^T J_f` and use the gradcheck defined above on this function.
Note that :math:`v` in this case is just a random vector with the same type as :math:`f(x)`.

The fast version of gradgrad check is implemented by using the fast version of gradcheck on that same function :math:`F`.
