// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_THREAD_POOL_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_THREAD_POOL_H

// evaluator for thread pool device
#ifdef EIGEN_USE_THREADS

namespace Eigen {
namespace internal {

template<typename LhsScalar, typename LhsMapper, typename Index>
struct packLhsArg {
  LhsScalar* blockA;
  const LhsMapper& lhs;
  const Index m_start;
  const Index k_start;
  const Index mc;
  const Index kc;
};

template<typename LhsScalar, typename RhsScalar, typename RhsMapper, typename OutputMapper, typename Index>
struct packRhsAndKernelArg {
  const std::vector<LhsScalar*>* blockAs;
  RhsScalar* blockB;
  const RhsMapper& rhs;
  OutputMapper& output;
  const Index m;
  const Index k;
  const Index n;
  const Index mc;
  const Index kc;
  const Index nc;
  const Index num_threads;
  const Index num_blockAs;
  const Index max_m;
  const Index k_block_idx;
  const Index m_block_idx;
  const Index n_block_idx;
  const Index m_blocks;
  const Index n_blocks;
  std::vector<Notification*>* kernel_notifications;
  const std::vector<Notification*>* lhs_notifications;
  const bool need_to_pack;
};

}  // end namespace internal


template<typename Indices, typename LeftArgType, typename RightArgType>
struct TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, ThreadPoolDevice> :
    public TensorContractionEvaluatorBase<TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, ThreadPoolDevice> > {

  typedef ThreadPoolDevice Device;

  typedef TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, Device> Self;
  typedef TensorContractionEvaluatorBase<Self> Base;

  typedef TensorContractionOp<Indices, LeftArgType, RightArgType> XprType;
  typedef typename internal::remove_const<typename XprType::Scalar>::type Scalar;
  typedef typename XprType::Packet Packet;
  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  enum {
    Layout = TensorEvaluator<LeftArgType, Device>::Layout,
  };

  // Most of the code is assuming that both input tensors are ColMajor. If the
  // inputs are RowMajor, we will "cheat" by swapping the LHS and RHS:
  // If we want to compute A * B = C, where A is LHS and B is RHS, the code
  // will pretend B is LHS and A is RHS.
  typedef typename internal::conditional<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), LeftArgType, RightArgType>::type EvalLeftArgType;
  typedef typename internal::conditional<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), RightArgType, LeftArgType>::type EvalRightArgType;

  static const int LDims =
      internal::array_size<typename TensorEvaluator<EvalLeftArgType, Device>::Dimensions>::value;
  static const int RDims =
      internal::array_size<typename TensorEvaluator<EvalRightArgType, Device>::Dimensions>::value;
  static const int ContractDims = internal::array_size<Indices>::value;

  typedef array<Index, LDims> left_dim_mapper_t;
  typedef array<Index, RDims> right_dim_mapper_t;

  typedef array<Index, ContractDims> contract_t;
  typedef array<Index, max_n_1<LDims - ContractDims>::size> left_nocontract_t;
  typedef array<Index, max_n_1<RDims - ContractDims>::size> right_nocontract_t;

  static const int NumDims = max_n_1<LDims + RDims - 2 * ContractDims>::size;

  typedef DSizes<Index, NumDims> Dimensions;

  // typedefs needed in evalTo
  typedef typename internal::remove_const<typename EvalLeftArgType::Scalar>::type LhsScalar;
  typedef typename internal::remove_const<typename EvalRightArgType::Scalar>::type RhsScalar;
  typedef typename internal::gebp_traits<LhsScalar, RhsScalar> Traits;

  typedef TensorEvaluator<EvalLeftArgType, Device> LeftEvaluator;
  typedef TensorEvaluator<EvalRightArgType, Device> RightEvaluator;

  TensorEvaluator(const XprType& op, const Device& device) :
      Base(op, device) {}

  template <bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered, int Alignment>
  void evalProduct(Scalar* buffer) const {
    if (this->m_j_size == 1) {
      this->template evalGemv<lhs_inner_dim_contiguous, rhs_inner_dim_contiguous, rhs_inner_dim_reordered, Alignment>(buffer);
      return;
    }

    evalGemm<lhs_inner_dim_contiguous, rhs_inner_dim_contiguous, rhs_inner_dim_reordered, Alignment>(buffer);
  }

  template <bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered, int Alignment>
  void evalGemm(Scalar* buffer) const {
    // columns in left side, rows in right side
    const Index k = this->m_k_size;

    // rows in left side
    const Index m = this->m_i_size;

    // columns in right side
    const Index n = this->m_j_size;

    // zero out the result buffer (which must be of size at least m * n * sizeof(Scalar)
    this->m_device.memset(buffer, 0, m * n * sizeof(Scalar));


    const int lhs_packet_size = internal::packet_traits<LhsScalar>::size;
    const int rhs_packet_size = internal::packet_traits<RhsScalar>::size;

    typedef internal::TensorContractionInputMapper<LhsScalar, Index, internal::Lhs,
                                                   LeftEvaluator, left_nocontract_t,
                                                   contract_t, lhs_packet_size,
                                                   lhs_inner_dim_contiguous,
                                                   false, Unaligned> LhsMapper;

    typedef internal::TensorContractionInputMapper<RhsScalar, Index, internal::Rhs,
                                                   RightEvaluator, right_nocontract_t,
                                                   contract_t, rhs_packet_size,
                                                   rhs_inner_dim_contiguous,
                                                   rhs_inner_dim_reordered, Unaligned> RhsMapper;

    typedef internal::blas_data_mapper<Scalar, Index, ColMajor> OutputMapper;

    // TODO: packing could be faster sometimes if we supported row major tensor mappers
    typedef internal::gemm_pack_lhs<LhsScalar, Index, typename LhsMapper::SubMapper, Traits::mr,
                                    Traits::LhsProgress, ColMajor> LhsPacker;
    typedef internal::gemm_pack_rhs<RhsScalar, Index, typename RhsMapper::SubMapper, Traits::nr, ColMajor> RhsPacker;

    // TODO: replace false, false with conjugate values?
    typedef internal::gebp_kernel<LhsScalar, RhsScalar, Index, OutputMapper,
                                  Traits::mr, Traits::nr, false, false> GebpKernel;

    typedef internal::packLhsArg<LhsScalar, LhsMapper, Index> packLArg;
    typedef internal::packRhsAndKernelArg<LhsScalar, RhsScalar, RhsMapper, OutputMapper, Index> packRKArg;

    // initialize data mappers
    LhsMapper lhs(this->m_leftImpl, this->m_left_nocontract_strides, this->m_i_strides,
                  this->m_left_contracting_strides, this->m_k_strides);

    RhsMapper rhs(this->m_rightImpl, this->m_right_nocontract_strides, this->m_j_strides,
                  this->m_right_contracting_strides, this->m_k_strides);

    OutputMapper output(buffer, m);

    // compute block sizes (which depend on number of threads)
    const Index num_threads = this->m_device.numThreads();
    Index mc = m;
    Index nc = n;
    Index kc = k;
    internal::computeProductBlockingSizes<LhsScalar,RhsScalar,1>(kc, mc, nc, num_threads);
    eigen_assert(mc <= m);
    eigen_assert(nc <= n);
    eigen_assert(kc <= k);

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
    const Index k_blocks = CEIL_DIV(k, kc);
    const Index n_blocks = CEIL_DIV(n, nc);
    const Index m_blocks = CEIL_DIV(m, mc);
    const Index sizeA = mc * kc;
    const Index sizeB = kc * nc;

    /*    cout << "m: " << m << " n: " << n << " k: " << k << endl;
    cout << "mc: " << mc << " nc: " << nc << " kc: " << kc << endl;
    cout << "m_blocks: " << m_blocks << " n_blocks: " << n_blocks << " k_blocks: " << k_blocks << endl;
    cout << "num threads: " << num_threads << endl;
    */

    // note: m_device.allocate should return 16 byte aligned pointers, but if blockA and blockB
    //       aren't 16 byte aligned segfaults will happen due to SIMD instructions
    // note: You can get away with allocating just a single blockA and offsets and meet the
    //       the alignment requirements with the assumption that
    //       (Traits::mr * sizeof(ResScalar)) % 16 == 0
    const Index numBlockAs = numext::mini(num_threads, m_blocks);
    std::vector<LhsScalar *> blockAs;
    blockAs.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
      blockAs.push_back(static_cast<LhsScalar *>(this->m_device.allocate(sizeA * sizeof(LhsScalar))));
    }

    // To circumvent alignment issues, I'm just going to separately allocate the memory for each thread
    // TODO: is this too much memory to allocate? This simplifies coding a lot, but is wasteful.
    //       Other options: (1) reuse memory when a thread finishes. con: tricky
    //                      (2) allocate block B memory in each thread. con: overhead
    std::vector<RhsScalar *> blockBs;
    blockBs.reserve(n_blocks);
    for (int i = 0; i < n_blocks; i++) {
      blockBs.push_back(static_cast<RhsScalar *>(this->m_device.allocate(sizeB * sizeof(RhsScalar))));
    }

    // lhs_notifications starts with all null Notifications
    std::vector<Notification*> lhs_notifications(num_threads, nullptr);

    // this should really be numBlockAs * n_blocks;
    const Index num_kernel_notifications = num_threads * n_blocks;
    std::vector<Notification*> kernel_notifications(num_kernel_notifications,
                                                    nullptr);

    for (Index k_block_idx = 0; k_block_idx < k_blocks; k_block_idx++) {
      const Index k_start = k_block_idx * kc;
      // make sure we don't overshoot right edge of left matrix
      const Index actual_kc = numext::mini(k_start + kc, k) - k_start;

      for (Index m_block_idx = 0; m_block_idx < m_blocks; m_block_idx += numBlockAs) {
        const Index num_blocks = numext::mini(m_blocks-m_block_idx, numBlockAs);

        for (Index mt_block_idx = m_block_idx; mt_block_idx < m_block_idx+num_blocks; mt_block_idx++) {
          const Index m_start = mt_block_idx * mc;
          const Index actual_mc = numext::mini(m_start + mc, m) - m_start;
          eigen_assert(actual_mc > 0);

          Index blockAId = (k_block_idx * m_blocks + mt_block_idx) % num_threads;

          for (int i = 0; i < n_blocks; ++i) {
            Index notification_id = (blockAId * n_blocks + i);
            // Wait for any current kernels using this slot to complete
            // before using it.
            if (kernel_notifications[notification_id]) {
              wait_until_ready(kernel_notifications[notification_id]);
              delete kernel_notifications[notification_id];
            }
            kernel_notifications[notification_id] = new Notification();
          }
          const packLArg arg = {
            blockAs[blockAId], // blockA
            lhs,        // lhs
            m_start,    // m
            k_start,    // k
            actual_mc,  // mc
            actual_kc,  // kc
          };

          // Delete any existing notification since we may be
          // replacing it.  The algorithm should ensure that there are
          // no existing waiters on this notification.
          delete lhs_notifications[blockAId];
          lhs_notifications[blockAId] =
          this->m_device.enqueue(&Self::packLhs<packLArg, LhsPacker>, arg);
        }

        // now start kernels.
        const Index m_base_start = m_block_idx * mc;
        const bool need_to_pack = m_block_idx == 0;

        for (Index n_block_idx = 0; n_block_idx < n_blocks; n_block_idx++) {
          const Index n_start = n_block_idx * nc;
          const Index actual_nc = numext::mini(n_start + nc, n) - n_start;

          // first make sure the previous kernels are all done before overwriting rhs. Also wait if
          // we're going to start new k. In both cases need_to_pack is true.
          if (need_to_pack) {
            for (Index i = num_blocks; i < num_threads; ++i) {
              Index blockAId = (k_block_idx * m_blocks + i + m_block_idx) % num_threads;
              Index future_id = (blockAId * n_blocks + n_block_idx);
              wait_until_ready(kernel_notifications[future_id]);
            }
          }

          packRKArg arg = {
            &blockAs, // blockA
            blockBs[n_block_idx], // blockB
            rhs,          // rhs
            output,       // output
            m_base_start, // m
            k_start,      // k
            n_start,      // n
            mc,           // mc
            actual_kc,    // kc
            actual_nc,    // nc
            num_threads,
            numBlockAs,
            m,
            k_block_idx,
            m_block_idx,
            n_block_idx, // n_block_idx
            m_blocks, // m_blocks
            n_blocks, // n_blocks
            &kernel_notifications, // kernel notifications
            &lhs_notifications,    // lhs notifications
            need_to_pack, // need_to_pack
          };

          // We asynchronously kick off this function, which ends up
          // notifying the appropriate kernel_notifications objects,
          // which this thread waits on before exiting.
          this->m_device.enqueueNoNotification(&Self::packRhsAndKernel<packRKArg, RhsPacker, GebpKernel>, arg);
        }
      }
    }

    // Make sure all the kernels are done.
    for (size_t i = 0; i < kernel_notifications.size(); ++i) {
      wait_until_ready(kernel_notifications[i]);
      delete kernel_notifications[i];
    }

    // No need to wait for lhs notifications since they should have
    // already been waited on.  Just clean them up.
    for (size_t i = 0; i < lhs_notifications.size(); ++i) {
      delete lhs_notifications[i];
    }

    // deallocate all of the memory for both A and B's
    for (size_t i = 0; i < blockAs.size(); i++) {
      this->m_device.deallocate(blockAs[i]);
    }
    for (size_t i = 0; i < blockBs.size(); i++) {
      this->m_device.deallocate(blockBs[i]);
    }

#undef CEIL_DIV
  }

  /*
   * Packs a LHS block of size (mt, kc) starting at lhs(m, k). Before packing
   * the LHS block, check that all of the kernels that worked on the same
   * mt_block_idx in the previous m_block are done.
   */
  template <typename packLArg, typename LhsPacker>
  static void packLhs(const packLArg arg) {
    // perform actual packing
    LhsPacker pack_lhs;
    pack_lhs(arg.blockA, arg.lhs.getSubMapper(arg.m_start, arg.k_start), arg.kc, arg.mc);
  }

  /*
   * Packs a RHS block of size (kc, nc) starting at (k, n) after checking that
   * all kernels in the previous block are done.
   * Then for each LHS future, we wait on the future and then call GEBP
   * on the area packed by the future (which starts at
   * blockA + future_idx * mt * kc) on the LHS and with the full packed
   * RHS block.
   * The output of this GEBP is written to output(m + i * mt, n).
   */
  template <typename packRKArg, typename RhsPacker, typename GebpKernel>
  static void packRhsAndKernel(packRKArg arg) {
    if (arg.need_to_pack) {
      RhsPacker pack_rhs;
      pack_rhs(arg.blockB, arg.rhs.getSubMapper(arg.k, arg.n), arg.kc, arg.nc);
    }

    GebpKernel gebp;
    for (Index mt_block_idx = 0; mt_block_idx < arg.num_blockAs; mt_block_idx++) {
      const Index m_base_start = arg.m + arg.mc*mt_block_idx;
      if (m_base_start < arg.max_m) {
        Index blockAId = (arg.k_block_idx * arg.m_blocks + mt_block_idx + arg.m_block_idx) % arg.num_threads;
        wait_until_ready((*arg.lhs_notifications)[blockAId]);
        const Index actual_mc = numext::mini(m_base_start + arg.mc, arg.max_m) - m_base_start;
        gebp(arg.output.getSubMapper(m_base_start, arg.n),
             (*arg.blockAs)[blockAId], arg.blockB,
             actual_mc, arg.kc, arg.nc, 1.0, -1, -1, 0, 0);

        // Notify that the kernel is done.
        const Index set_idx = blockAId * arg.n_blocks + arg.n_block_idx;
        (*arg.kernel_notifications)[set_idx]->Notify();
      }
    }
  }
};

} // end namespace Eigen

#endif  // EIGEN_USE_THREADS
#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_THREAD_POOL_H
