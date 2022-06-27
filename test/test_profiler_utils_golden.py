# Owner(s): ["oncall: profiler"]
'''
        def garbage_code(x):
            for i in range(5):
                x[0, i] = i

        x = torch.ones((4096, 4096), device="cuda")
        x = x @ x
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True,
                     with_stack=True) as prof:
            for _ in range(5):
                x = x @ x
            garbage_code(x)
            for _ in range(5):
                x = x @ x

        kineto_event_string = json.dumps(
            [{
                '_name': e.name(),
                '_start_us': e.start_us(),
                '_duration_us': e.duration_us(),
                '_linked_correlation_id': e.linked_correlation_id(),
                '_device_type': 1 if e.device_type() == DeviceType.CUDA else 0
            } for e in prof.profiler.kineto_results.events()],
            indent=4)

        def EventTreeDFS(event_tree):
            from collections import deque
            stack = deque(event_tree)
            while stack:
                curr_event = stack.pop()
                yield curr_event
                for child_event in curr_event.children:
                    stack.append(child_event)

        profiler_event_string = json.dumps(
            [{
                '_name': e.name(),
                'id': e.id,
                'start_time_ns': e.start_time_ns,
                'duration_time_ns': e.duration_time_ns,
                'correlation_id': e.correlation_id,
                'children': [child.id for child in e.children],
                'parent': e.parent.id if e.parent else None
            } for e in EventTreeDFS(prof.profiler.kineto_results.experimental_event_tree())],
            indent=4)
'''

kineto_event_string = '''\
[
    {
        "_name": "aten::matmul",
        "_start_us": 1656363168761401,
        "_duration_us": 1153,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::mm",
        "_start_us": 1656363168761406,
        "_duration_us": 1145,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::matmul",
        "_start_us": 1656363168762572,
        "_duration_us": 32,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::mm",
        "_start_us": 1656363168762574,
        "_duration_us": 29,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::matmul",
        "_start_us": 1656363168762607,
        "_duration_us": 27,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::mm",
        "_start_us": 1656363168762607,
        "_duration_us": 26,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::matmul",
        "_start_us": 1656363168762636,
        "_duration_us": 18,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::mm",
        "_start_us": 1656363168762637,
        "_duration_us": 16,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::matmul",
        "_start_us": 1656363168762656,
        "_duration_us": 17,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::mm",
        "_start_us": 1656363168762657,
        "_duration_us": 16,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168764614,
        "_duration_us": 7674,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168770373,
        "_duration_us": 4,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168772291,
        "_duration_us": 664,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168772953,
        "_duration_us": 1,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::copy_",
        "_start_us": 1656363168772960,
        "_duration_us": 42868,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168815847,
        "_duration_us": 4,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168815849,
        "_duration_us": 2,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168815852,
        "_duration_us": 2,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168815854,
        "_duration_us": 0,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::copy_",
        "_start_us": 1656363168815856,
        "_duration_us": 29,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168815890,
        "_duration_us": 3,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168815892,
        "_duration_us": 0,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168815893,
        "_duration_us": 2,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168815894,
        "_duration_us": 1,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::copy_",
        "_start_us": 1656363168815896,
        "_duration_us": 20,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168815920,
        "_duration_us": 3,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168815922,
        "_duration_us": 0,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168815923,
        "_duration_us": 2,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168815924,
        "_duration_us": 0,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::copy_",
        "_start_us": 1656363168815926,
        "_duration_us": 20,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168815950,
        "_duration_us": 3,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168815952,
        "_duration_us": 0,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168815954,
        "_duration_us": 1,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168815955,
        "_duration_us": 0,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::copy_",
        "_start_us": 1656363168815956,
        "_duration_us": 19,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168815980,
        "_duration_us": 2,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168815981,
        "_duration_us": 0,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168815982,
        "_duration_us": 2,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168815983,
        "_duration_us": 0,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::copy_",
        "_start_us": 1656363168815985,
        "_duration_us": 19,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168816008,
        "_duration_us": 2,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168816009,
        "_duration_us": 1,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168816011,
        "_duration_us": 1,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168816012,
        "_duration_us": 0,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::copy_",
        "_start_us": 1656363168816013,
        "_duration_us": 20,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168816037,
        "_duration_us": 3,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168816039,
        "_duration_us": 0,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168816040,
        "_duration_us": 2,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168816041,
        "_duration_us": 0,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::copy_",
        "_start_us": 1656363168816043,
        "_duration_us": 19,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168816066,
        "_duration_us": 2,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168816067,
        "_duration_us": 1,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168816069,
        "_duration_us": 2,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168816070,
        "_duration_us": 0,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::copy_",
        "_start_us": 1656363168816071,
        "_duration_us": 20,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168816095,
        "_duration_us": 2,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168816096,
        "_duration_us": 0,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::select",
        "_start_us": 1656363168816098,
        "_duration_us": 1,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::as_strided",
        "_start_us": 1656363168816098,
        "_duration_us": 1,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::copy_",
        "_start_us": 1656363168816100,
        "_duration_us": 19,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::matmul",
        "_start_us": 1656363168816130,
        "_duration_us": 56,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::mm",
        "_start_us": 1656363168816132,
        "_duration_us": 53,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::matmul",
        "_start_us": 1656363168816203,
        "_duration_us": 22,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::mm",
        "_start_us": 1656363168816204,
        "_duration_us": 21,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::matmul",
        "_start_us": 1656363168816228,
        "_duration_us": 28,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::mm",
        "_start_us": 1656363168816228,
        "_duration_us": 27,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::matmul",
        "_start_us": 1656363168816258,
        "_duration_us": 18,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::mm",
        "_start_us": 1656363168816258,
        "_duration_us": 17,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::matmul",
        "_start_us": 1656363168816278,
        "_duration_us": 17,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "aten::mm",
        "_start_us": 1656363168816278,
        "_duration_us": 16,
        "_linked_correlation_id": 0,
        "_device_type": 0
    },
    {
        "_name": "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        "_start_us": 1656363168761629,
        "_duration_us": 2,
        "_linked_correlation_id": 3074,
        "_device_type": 0
    },
    {
        "_name": "ampere_sgemm_128x64_nn",
        "_start_us": 1656363168769381,
        "_duration_us": 9286,
        "_linked_correlation_id": 3074,
        "_device_type": 1
    },
    {
        "_name": "cudaLaunchKernel",
        "_start_us": 1656363168761635,
        "_duration_us": 912,
        "_linked_correlation_id": 3074,
        "_device_type": 0
    },
    {
        "_name": "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        "_start_us": 1656363168762592,
        "_duration_us": 1,
        "_linked_correlation_id": 3076,
        "_device_type": 0
    },
    {
        "_name": "ampere_sgemm_128x64_nn",
        "_start_us": 1656363168778669,
        "_duration_us": 9283,
        "_linked_correlation_id": 3076,
        "_device_type": 1
    },
    {
        "_name": "cudaLaunchKernel",
        "_start_us": 1656363168762594,
        "_duration_us": 7,
        "_linked_correlation_id": 3076,
        "_device_type": 0
    },
    {
        "_name": "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        "_start_us": 1656363168762625,
        "_duration_us": 0,
        "_linked_correlation_id": 3078,
        "_device_type": 0
    },
    {
        "_name": "ampere_sgemm_128x64_nn",
        "_start_us": 1656363168787953,
        "_duration_us": 9285,
        "_linked_correlation_id": 3078,
        "_device_type": 1
    },
    {
        "_name": "cudaLaunchKernel",
        "_start_us": 1656363168762627,
        "_duration_us": 4,
        "_linked_correlation_id": 3078,
        "_device_type": 0
    },
    {
        "_name": "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        "_start_us": 1656363168762645,
        "_duration_us": 0,
        "_linked_correlation_id": 3080,
        "_device_type": 0
    },
    {
        "_name": "ampere_sgemm_128x64_nn",
        "_start_us": 1656363168797240,
        "_duration_us": 9285,
        "_linked_correlation_id": 3080,
        "_device_type": 1
    },
    {
        "_name": "cudaLaunchKernel",
        "_start_us": 1656363168762647,
        "_duration_us": 5,
        "_linked_correlation_id": 3080,
        "_device_type": 0
    },
    {
        "_name": "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        "_start_us": 1656363168762665,
        "_duration_us": 0,
        "_linked_correlation_id": 3082,
        "_device_type": 0
    },
    {
        "_name": "ampere_sgemm_128x64_nn",
        "_start_us": 1656363168806527,
        "_duration_us": 9284,
        "_linked_correlation_id": 3082,
        "_device_type": 1
    },
    {
        "_name": "cudaLaunchKernel",
        "_start_us": 1656363168762666,
        "_duration_us": 5,
        "_linked_correlation_id": 3082,
        "_device_type": 0
    },
    {
        "_name": "Memcpy HtoD (Pageable -> Device)",
        "_start_us": 1656363168815814,
        "_duration_us": 2,
        "_linked_correlation_id": 3087,
        "_device_type": 1
    },
    {
        "_name": "cudaMemcpyAsync",
        "_start_us": 1656363168772979,
        "_duration_us": 17,
        "_linked_correlation_id": 3087,
        "_device_type": 0
    },
    {
        "_name": "cudaStreamSynchronize",
        "_start_us": 1656363168772998,
        "_duration_us": 42827,
        "_linked_correlation_id": 3087,
        "_device_type": 0
    },
    {
        "_name": "Memcpy HtoD (Pageable -> Device)",
        "_start_us": 1656363168815876,
        "_duration_us": 2,
        "_linked_correlation_id": 3092,
        "_device_type": 1
    },
    {
        "_name": "cudaMemcpyAsync",
        "_start_us": 1656363168815863,
        "_duration_us": 6,
        "_linked_correlation_id": 3092,
        "_device_type": 0
    },
    {
        "_name": "cudaStreamSynchronize",
        "_start_us": 1656363168815870,
        "_duration_us": 14,
        "_linked_correlation_id": 3092,
        "_device_type": 0
    },
    {
        "_name": "Memcpy HtoD (Pageable -> Device)",
        "_start_us": 1656363168815907,
        "_duration_us": 2,
        "_linked_correlation_id": 3097,
        "_device_type": 1
    },
    {
        "_name": "cudaMemcpyAsync",
        "_start_us": 1656363168815899,
        "_duration_us": 3,
        "_linked_correlation_id": 3097,
        "_device_type": 0
    },
    {
        "_name": "cudaStreamSynchronize",
        "_start_us": 1656363168815902,
        "_duration_us": 13,
        "_linked_correlation_id": 3097,
        "_device_type": 0
    },
    {
        "_name": "Memcpy HtoD (Pageable -> Device)",
        "_start_us": 1656363168815936,
        "_duration_us": 2,
        "_linked_correlation_id": 3102,
        "_device_type": 1
    },
    {
        "_name": "cudaMemcpyAsync",
        "_start_us": 1656363168815928,
        "_duration_us": 3,
        "_linked_correlation_id": 3102,
        "_device_type": 0
    },
    {
        "_name": "cudaStreamSynchronize",
        "_start_us": 1656363168815931,
        "_duration_us": 13,
        "_linked_correlation_id": 3102,
        "_device_type": 0
    },
    {
        "_name": "Memcpy HtoD (Pageable -> Device)",
        "_start_us": 1656363168815966,
        "_duration_us": 2,
        "_linked_correlation_id": 3107,
        "_device_type": 1
    },
    {
        "_name": "cudaMemcpyAsync",
        "_start_us": 1656363168815958,
        "_duration_us": 3,
        "_linked_correlation_id": 3107,
        "_device_type": 0
    },
    {
        "_name": "cudaStreamSynchronize",
        "_start_us": 1656363168815962,
        "_duration_us": 12,
        "_linked_correlation_id": 3107,
        "_device_type": 0
    },
    {
        "_name": "Memcpy HtoD (Pageable -> Device)",
        "_start_us": 1656363168815995,
        "_duration_us": 2,
        "_linked_correlation_id": 3112,
        "_device_type": 1
    },
    {
        "_name": "cudaMemcpyAsync",
        "_start_us": 1656363168815987,
        "_duration_us": 3,
        "_linked_correlation_id": 3112,
        "_device_type": 0
    },
    {
        "_name": "cudaStreamSynchronize",
        "_start_us": 1656363168815990,
        "_duration_us": 12,
        "_linked_correlation_id": 3112,
        "_device_type": 0
    },
    {
        "_name": "Memcpy HtoD (Pageable -> Device)",
        "_start_us": 1656363168816024,
        "_duration_us": 2,
        "_linked_correlation_id": 3117,
        "_device_type": 1
    },
    {
        "_name": "cudaMemcpyAsync",
        "_start_us": 1656363168816016,
        "_duration_us": 3,
        "_linked_correlation_id": 3117,
        "_device_type": 0
    },
    {
        "_name": "cudaStreamSynchronize",
        "_start_us": 1656363168816019,
        "_duration_us": 12,
        "_linked_correlation_id": 3117,
        "_device_type": 0
    },
    {
        "_name": "Memcpy HtoD (Pageable -> Device)",
        "_start_us": 1656363168816053,
        "_duration_us": 2,
        "_linked_correlation_id": 3122,
        "_device_type": 1
    },
    {
        "_name": "cudaMemcpyAsync",
        "_start_us": 1656363168816045,
        "_duration_us": 3,
        "_linked_correlation_id": 3122,
        "_device_type": 0
    },
    {
        "_name": "cudaStreamSynchronize",
        "_start_us": 1656363168816048,
        "_duration_us": 13,
        "_linked_correlation_id": 3122,
        "_device_type": 0
    },
    {
        "_name": "Memcpy HtoD (Pageable -> Device)",
        "_start_us": 1656363168816082,
        "_duration_us": 2,
        "_linked_correlation_id": 3127,
        "_device_type": 1
    },
    {
        "_name": "cudaMemcpyAsync",
        "_start_us": 1656363168816074,
        "_duration_us": 3,
        "_linked_correlation_id": 3127,
        "_device_type": 0
    },
    {
        "_name": "cudaStreamSynchronize",
        "_start_us": 1656363168816077,
        "_duration_us": 13,
        "_linked_correlation_id": 3127,
        "_device_type": 0
    },
    {
        "_name": "Memcpy HtoD (Pageable -> Device)",
        "_start_us": 1656363168816110,
        "_duration_us": 2,
        "_linked_correlation_id": 3132,
        "_device_type": 1
    },
    {
        "_name": "cudaMemcpyAsync",
        "_start_us": 1656363168816102,
        "_duration_us": 3,
        "_linked_correlation_id": 3132,
        "_device_type": 0
    },
    {
        "_name": "cudaStreamSynchronize",
        "_start_us": 1656363168816105,
        "_duration_us": 13,
        "_linked_correlation_id": 3132,
        "_device_type": 0
    },
    {
        "_name": "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        "_start_us": 1656363168816165,
        "_duration_us": 1,
        "_linked_correlation_id": 3134,
        "_device_type": 0
    },
    {
        "_name": "ampere_sgemm_128x64_nn",
        "_start_us": 1656363168816185,
        "_duration_us": 9282,
        "_linked_correlation_id": 3134,
        "_device_type": 1
    },
    {
        "_name": "cudaLaunchKernel",
        "_start_us": 1656363168816169,
        "_duration_us": 13,
        "_linked_correlation_id": 3134,
        "_device_type": 0
    },
    {
        "_name": "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        "_start_us": 1656363168816215,
        "_duration_us": 0,
        "_linked_correlation_id": 3136,
        "_device_type": 0
    },
    {
        "_name": "ampere_sgemm_128x64_nn",
        "_start_us": 1656363168825469,
        "_duration_us": 9283,
        "_linked_correlation_id": 3136,
        "_device_type": 1
    },
    {
        "_name": "cudaLaunchKernel",
        "_start_us": 1656363168816217,
        "_duration_us": 6,
        "_linked_correlation_id": 3136,
        "_device_type": 0
    },
    {
        "_name": "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        "_start_us": 1656363168816247,
        "_duration_us": 0,
        "_linked_correlation_id": 3138,
        "_device_type": 0
    },
    {
        "_name": "ampere_sgemm_128x64_nn",
        "_start_us": 1656363168834754,
        "_duration_us": 9282,
        "_linked_correlation_id": 3138,
        "_device_type": 1
    },
    {
        "_name": "cudaLaunchKernel",
        "_start_us": 1656363168816248,
        "_duration_us": 5,
        "_linked_correlation_id": 3138,
        "_device_type": 0
    },
    {
        "_name": "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        "_start_us": 1656363168816267,
        "_duration_us": 0,
        "_linked_correlation_id": 3140,
        "_device_type": 0
    },
    {
        "_name": "ampere_sgemm_128x64_nn",
        "_start_us": 1656363168844038,
        "_duration_us": 9288,
        "_linked_correlation_id": 3140,
        "_device_type": 1
    },
    {
        "_name": "cudaLaunchKernel",
        "_start_us": 1656363168816269,
        "_duration_us": 4,
        "_linked_correlation_id": 3140,
        "_device_type": 0
    },
    {
        "_name": "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        "_start_us": 1656363168816287,
        "_duration_us": 0,
        "_linked_correlation_id": 3142,
        "_device_type": 0
    },
    {
        "_name": "ampere_sgemm_128x64_nn",
        "_start_us": 1656363168853328,
        "_duration_us": 9284,
        "_linked_correlation_id": 3142,
        "_device_type": 1
    },
    {
        "_name": "cudaLaunchKernel",
        "_start_us": 1656363168816288,
        "_duration_us": 4,
        "_linked_correlation_id": 3142,
        "_device_type": 0
    },
    {
        "_name": "cudaDeviceSynchronize",
        "_start_us": 1656363168816413,
        "_duration_us": 46206,
        "_linked_correlation_id": 0,
        "_device_type": 0
    }
]'''

profiler_event_string = '''\
[
    {
        "_name": "test_profiler.py(3344): <module>",
        "id": 94875115641792,
        "start_time_ns": 1656363168760410721,
        "duration_time_ns": 7567008868094365087,
        "correlation_id": 0,
        "children": [
            94875144468128
        ],
        "parent": null
    },
    {
        "_name": "torch/testing/_internal/common_utils.py(697): run_tests",
        "id": 94875144468128,
        "start_time_ns": 1656363168760417415,
        "duration_time_ns": 7567008868094358393,
        "correlation_id": 0,
        "children": [
            94875144468592
        ],
        "parent": 94875115641792
    },
    {
        "_name": "unittest/main.py(101): __init__",
        "id": 94875144468592,
        "start_time_ns": 1656363168760418481,
        "duration_time_ns": 7567008868094357327,
        "correlation_id": 0,
        "children": [
            94875144468976
        ],
        "parent": 94875144468128
    },
    {
        "_name": "unittest/main.py(271): runTests",
        "id": 94875144468976,
        "start_time_ns": 1656363168760419783,
        "duration_time_ns": 7567008868094356025,
        "correlation_id": 0,
        "children": [
            94875145711840
        ],
        "parent": 94875144468592
    },
    {
        "_name": "unittest/runner.py(184): run",
        "id": 94875145711840,
        "start_time_ns": 1656363168760420484,
        "duration_time_ns": 7567008868094355324,
        "correlation_id": 0,
        "children": [
            94875145712256
        ],
        "parent": 94875144468976
    },
    {
        "_name": "unittest/suite.py(84): __call__",
        "id": 94875145712256,
        "start_time_ns": 1656363168760421797,
        "duration_time_ns": 7567008868094354011,
        "correlation_id": 0,
        "children": [
            94875145712672
        ],
        "parent": 94875145711840
    },
    {
        "_name": "unittest/suite.py(122): run",
        "id": 94875145712672,
        "start_time_ns": 1656363168760423088,
        "duration_time_ns": 7567008868094352720,
        "correlation_id": 0,
        "children": [
            94875145713088
        ],
        "parent": 94875145712256
    },
    {
        "_name": "unittest/suite.py(84): __call__",
        "id": 94875145713088,
        "start_time_ns": 1656363168760424016,
        "duration_time_ns": 7567008868094351792,
        "correlation_id": 0,
        "children": [
            94875145713504
        ],
        "parent": 94875145712672
    },
    {
        "_name": "unittest/suite.py(122): run",
        "id": 94875145713504,
        "start_time_ns": 1656363168760424177,
        "duration_time_ns": 7567008868094351631,
        "correlation_id": 0,
        "children": [
            94875145713920
        ],
        "parent": 94875145713088
    },
    {
        "_name": "unittest/case.py(651): __call__",
        "id": 94875145713920,
        "start_time_ns": 1656363168760424312,
        "duration_time_ns": 7567008868094351496,
        "correlation_id": 0,
        "children": [
            94875145483520
        ],
        "parent": 94875145713504
    },
    {
        "_name": "torch/testing/_internal/common_utils.py(1886): run",
        "id": 94875145483520,
        "start_time_ns": 1656363168760427569,
        "duration_time_ns": 7567008868094348239,
        "correlation_id": 0,
        "children": [
            94875145483936
        ],
        "parent": 94875145713920
    },
    {
        "_name": "torch/testing/_internal/common_utils.py(1829): _run_with_retry",
        "id": 94875145483936,
        "start_time_ns": 1656363168760429072,
        "duration_time_ns": 7567008868094346736,
        "correlation_id": 0,
        "children": [
            94875145484352
        ],
        "parent": 94875145483520
    },
    {
        "_name": "unittest/case.py(592): run",
        "id": 94875145484352,
        "start_time_ns": 1656363168760429743,
        "duration_time_ns": 7567008868094346065,
        "correlation_id": 0,
        "children": [
            94875145484768
        ],
        "parent": 94875145483936
    },
    {
        "_name": "unittest/case.py(550): _callTestMethod",
        "id": 94875145484768,
        "start_time_ns": 1656363168760430244,
        "duration_time_ns": 7567008868094345564,
        "correlation_id": 0,
        "children": [
            94875145485184
        ],
        "parent": 94875145484352
    },
    {
        "_name": "test_profiler.py(1374): test_utils_get_optimizable_events",
        "id": 94875145485184,
        "start_time_ns": 1656363168760430825,
        "duration_time_ns": 7567008868094344983,
        "correlation_id": 0,
        "children": [
            94875145485600,
            94873905411152,
            94873875387024,
            94873869202528,
            94873884074288,
            94875115652976,
            94875145537968,
            94875144437392,
            94875144438736,
            94875144440080,
            94875144443200,
            94875144444544,
            94875145538384
        ],
        "parent": 94875145484768
    },
    {
        "_name": "torch/profiler/profiler.py(473): __exit__",
        "id": 94875145538384,
        "start_time_ns": 1656363168816299194,
        "duration_time_ns": 7567008868038476614,
        "correlation_id": 0,
        "children": [
            94875145538800
        ],
        "parent": 94875145485184
    },
    {
        "_name": "torch/profiler/profiler.py(482): stop",
        "id": 94875145538800,
        "start_time_ns": 1656363168816304022,
        "duration_time_ns": 7567008868038471786,
        "correlation_id": 0,
        "children": [
            94875145539216
        ],
        "parent": 94875145538384
    },
    {
        "_name": "torch/profiler/profiler.py(509): _transit_action",
        "id": 94875145539216,
        "start_time_ns": 1656363168816307083,
        "duration_time_ns": 7567008868038468725,
        "correlation_id": 0,
        "children": [
            94875142519344,
            94875145540048
        ],
        "parent": 94875145538800
    },
    {
        "_name": "torch/profiler/profiler.py(115): stop_trace",
        "id": 94875145540048,
        "start_time_ns": 1656363168816331030,
        "duration_time_ns": 7567008868038444778,
        "correlation_id": 0,
        "children": [
            94875145540464
        ],
        "parent": 94875145539216
    },
    {
        "_name": "torch/autograd/profiler.py(207): __exit__",
        "id": 94875145540464,
        "start_time_ns": 1656363168816333575,
        "duration_time_ns": 7567008868038442233,
        "correlation_id": 0,
        "children": [
            94875145540880
        ],
        "parent": 94875145540048
    },
    {
        "_name": "torch/cuda/__init__.py(486): synchronize",
        "id": 94875145540880,
        "start_time_ns": 1656363168816337979,
        "duration_time_ns": 7567008868038437829,
        "correlation_id": 0,
        "children": [
            94875145541296,
            94875145542128,
            94875142515152,
            94875115620704,
            94875142518064
        ],
        "parent": 94875145540464
    },
    {
        "_name": "torch/cuda/__init__.py(281): __exit__",
        "id": 94875142518064,
        "start_time_ns": 1656363168862621095,
        "duration_time_ns": 7567008867992154713,
        "correlation_id": 0,
        "children": [
            94875115621120
        ],
        "parent": 94875145540880
    },
    {
        "_name": "<built-in method _disable_profiler of PyCapsule object at 0x7f8c84b00060>",
        "id": 94875115621120,
        "start_time_ns": 1656363168862629827,
        "duration_time_ns": 7567008867992145981,
        "correlation_id": 0,
        "children": [],
        "parent": 94875142518064
    },
    {
        "_name": "<built-in function _cuda_synchronize>",
        "id": 94875115620704,
        "start_time_ns": 1656363168816412203,
        "duration_time_ns": 46207846,
        "correlation_id": 0,
        "children": [],
        "parent": 94875145540880
    },
    {
        "_name": "torch/cuda/__init__.py(272): __enter__",
        "id": 94875142515152,
        "start_time_ns": 1656363168816404887,
        "duration_time_ns": 4818,
        "correlation_id": 0,
        "children": [
            94875142515568,
            94875142516816,
            94875142517232
        ],
        "parent": 94875145540880
    },
    {
        "_name": "torch/cuda/__init__.py(191): _lazy_init",
        "id": 94875142517232,
        "start_time_ns": 1656363168816409040,
        "duration_time_ns": 600,
        "correlation_id": 0,
        "children": [
            94875142517648
        ],
        "parent": 94875142515152
    },
    {
        "_name": "torch/cuda/__init__.py(149): is_initialized",
        "id": 94875142517648,
        "start_time_ns": 1656363168816409401,
        "duration_time_ns": 197,
        "correlation_id": 0,
        "children": [
            94875115620288
        ],
        "parent": 94875142517232
    },
    {
        "_name": "<built-in function _cuda_isInBadFork>",
        "id": 94875115620288,
        "start_time_ns": 1656363168816409497,
        "duration_time_ns": 58,
        "correlation_id": 0,
        "children": [],
        "parent": 94875142517648
    },
    {
        "_name": "torch/_jit_internal.py(982): is_scripting",
        "id": 94875142516816,
        "start_time_ns": 1656363168816408393,
        "duration_time_ns": 442,
        "correlation_id": 0,
        "children": [],
        "parent": 94875142515152
    },
    {
        "_name": "torch/cuda/__init__.py(480): current_device",
        "id": 94875142515568,
        "start_time_ns": 1656363168816405958,
        "duration_time_ns": 1853,
        "correlation_id": 0,
        "children": [
            94875142515984,
            94875142527616
        ],
        "parent": 94875142515152
    },
    {
        "_name": "<built-in function _cuda_getDevice>",
        "id": 94875142527616,
        "start_time_ns": 1656363168816407055,
        "duration_time_ns": 726,
        "correlation_id": 0,
        "children": [],
        "parent": 94875142515568
    },
    {
        "_name": "torch/cuda/__init__.py(191): _lazy_init",
        "id": 94875142515984,
        "start_time_ns": 1656363168816406355,
        "duration_time_ns": 466,
        "correlation_id": 0,
        "children": [
            94875142516400
        ],
        "parent": 94875142515568
    },
    {
        "_name": "torch/cuda/__init__.py(149): is_initialized",
        "id": 94875142516400,
        "start_time_ns": 1656363168816406525,
        "duration_time_ns": 248,
        "correlation_id": 0,
        "children": [
            94875142527200
        ],
        "parent": 94875142515984
    },
    {
        "_name": "<built-in function _cuda_isInBadFork>",
        "id": 94875142527200,
        "start_time_ns": 1656363168816406675,
        "duration_time_ns": 66,
        "correlation_id": 0,
        "children": [],
        "parent": 94875142516400
    },
    {
        "_name": "torch/cuda/__init__.py(268): __init__",
        "id": 94875145542128,
        "start_time_ns": 1656363168816349277,
        "duration_time_ns": 54066,
        "correlation_id": 0,
        "children": [
            94875145542544
        ],
        "parent": 94875145540880
    },
    {
        "_name": "torch/cuda/_utils.py(7): _get_device_index",
        "id": 94875145542544,
        "start_time_ns": 1656363168816354065,
        "duration_time_ns": 46961,
        "correlation_id": 0,
        "children": [
            94875142520560,
            94875142520976,
            94875145542960,
            94875142521392,
            94875145543376
        ],
        "parent": 94875145542128
    },
    {
        "_name": "torch/_utils.py(521): _get_device_index",
        "id": 94875145543376,
        "start_time_ns": 1656363168816364479,
        "duration_time_ns": 36382,
        "correlation_id": 0,
        "children": [
            94875142521808,
            94875142522224,
            94875142522640,
            94875145543792,
            94875142511824
        ],
        "parent": 94875145542544
    },
    {
        "_name": "torch/_utils.py(497): _get_current_device_index",
        "id": 94875142511824,
        "start_time_ns": 1656363168816375090,
        "duration_time_ns": 25635,
        "correlation_id": 0,
        "children": [
            94875142512240
        ],
        "parent": 94875145543376
    },
    {
        "_name": "torch/_utils.py(487): _get_device_attr",
        "id": 94875142512240,
        "start_time_ns": 1656363168816378065,
        "duration_time_ns": 22386,
        "correlation_id": 0,
        "children": [
            94875142512656,
            94875142525952,
            94875142513488
        ],
        "parent": 94875142511824
    },
    {
        "_name": "torch/_utils.py(499): <lambda>",
        "id": 94875142513488,
        "start_time_ns": 1656363168816391717,
        "duration_time_ns": 8574,
        "correlation_id": 0,
        "children": [
            94875142513904
        ],
        "parent": 94875142512240
    },
    {
        "_name": "torch/cuda/__init__.py(480): current_device",
        "id": 94875142513904,
        "start_time_ns": 1656363168816393074,
        "duration_time_ns": 7097,
        "correlation_id": 0,
        "children": [
            94875142514320,
            94875142526784
        ],
        "parent": 94875142513488
    },
    {
        "_name": "<built-in function _cuda_getDevice>",
        "id": 94875142526784,
        "start_time_ns": 1656363168816397007,
        "duration_time_ns": 3116,
        "correlation_id": 0,
        "children": [],
        "parent": 94875142513904
    },
    {
        "_name": "torch/cuda/__init__.py(191): _lazy_init",
        "id": 94875142514320,
        "start_time_ns": 1656363168816394245,
        "duration_time_ns": 1041,
        "correlation_id": 0,
        "children": [
            94875142514736
        ],
        "parent": 94875142513904
    },
    {
        "_name": "torch/cuda/__init__.py(149): is_initialized",
        "id": 94875142514736,
        "start_time_ns": 1656363168816394823,
        "duration_time_ns": 388,
        "correlation_id": 0,
        "children": [
            94875142526368
        ],
        "parent": 94875142514320
    },
    {
        "_name": "<built-in function _cuda_isInBadFork>",
        "id": 94875142526368,
        "start_time_ns": 1656363168816395034,
        "duration_time_ns": 109,
        "correlation_id": 0,
        "children": [],
        "parent": 94875142514736
    },
    {
        "_name": "<built-in method get of dict object at 0x7f8c7c18edc0>",
        "id": 94875142525952,
        "start_time_ns": 1656363168816390887,
        "duration_time_ns": 261,
        "correlation_id": 0,
        "children": [],
        "parent": 94875142512240
    },
    {
        "_name": "torch/_utils.py(478): _get_available_device_type",
        "id": 94875142512656,
        "start_time_ns": 1656363168816383127,
        "duration_time_ns": 6624,
        "correlation_id": 0,
        "children": [
            94875142513072
        ],
        "parent": 94875142512240
    },
    {
        "_name": "torch/cuda/__init__.py(77): is_available",
        "id": 94875142513072,
        "start_time_ns": 1656363168816384727,
        "duration_time_ns": 4858,
        "correlation_id": 0,
        "children": [
            94875142525120,
            94875142525536
        ],
        "parent": 94875142512656
    },
    {
        "_name": "<built-in function _cuda_getDeviceCount>",
        "id": 94875142525536,
        "start_time_ns": 1656363168816388458,
        "duration_time_ns": 903,
        "correlation_id": 0,
        "children": [],
        "parent": 94875142513072
    },
    {
        "_name": "<built-in function hasattr>",
        "id": 94875142525120,
        "start_time_ns": 1656363168816386687,
        "duration_time_ns": 711,
        "correlation_id": 0,
        "children": [],
        "parent": 94875142513072
    },
    {
        "_name": "torch/_jit_internal.py(982): is_scripting",
        "id": 94875145543792,
        "start_time_ns": 1656363168816373327,
        "duration_time_ns": 1172,
        "correlation_id": 0,
        "children": [],
        "parent": 94875145543376
    },
    {
        "_name": "<built-in function isinstance>",
        "id": 94875142522640,
        "start_time_ns": 1656363168816372934,
        "duration_time_ns": 44,
        "correlation_id": 0,
        "children": [],
        "parent": 94875145543376
    },
    {
        "_name": "<built-in function isinstance>",
        "id": 94875142522224,
        "start_time_ns": 1656363168816372052,
        "duration_time_ns": 44,
        "correlation_id": 0,
        "children": [],
        "parent": 94875145543376
    },
    {
        "_name": "<built-in function isinstance>",
        "id": 94875142521808,
        "start_time_ns": 1656363168816371387,
        "duration_time_ns": 59,
        "correlation_id": 0,
        "children": [],
        "parent": 94875145543376
    },
    {
        "_name": "<built-in function isinstance>",
        "id": 94875142521392,
        "start_time_ns": 1656363168816363013,
        "duration_time_ns": 46,
        "correlation_id": 0,
        "children": [],
        "parent": 94875145542544
    },
    {
        "_name": "torch/_jit_internal.py(982): is_scripting",
        "id": 94875145542960,
        "start_time_ns": 1656363168816361398,
        "duration_time_ns": 975,
        "correlation_id": 0,
        "children": [],
        "parent": 94875145542544
    },
    {
        "_name": "<built-in function isinstance>",
        "id": 94875142520976,
        "start_time_ns": 1656363168816358907,
        "duration_time_ns": 87,
        "correlation_id": 0,
        "children": [],
        "parent": 94875145542544
    },
    {
        "_name": "<built-in function isinstance>",
        "id": 94875142520560,
        "start_time_ns": 1656363168816356621,
        "duration_time_ns": 817,
        "correlation_id": 0,
        "children": [],
        "parent": 94875145542544
    },
    {
        "_name": "torch/cuda/__init__.py(191): _lazy_init",
        "id": 94875145541296,
        "start_time_ns": 1656363168816340275,
        "duration_time_ns": 6552,
        "correlation_id": 0,
        "children": [
            94875145541712
        ],
        "parent": 94875145540880
    },
    {
        "_name": "torch/cuda/__init__.py(149): is_initialized",
        "id": 94875145541712,
        "start_time_ns": 1656363168816341933,
        "duration_time_ns": 4718,
        "correlation_id": 0,
        "children": [
            94875142520176
        ],
        "parent": 94875145541296
    },
    {
        "_name": "<built-in function _cuda_isInBadFork>",
        "id": 94875142520176,
        "start_time_ns": 1656363168816345771,
        "duration_time_ns": 702,
        "correlation_id": 0,
        "children": [],
        "parent": 94875145541712
    },
    {
        "_name": "<built-in method get of dict object at 0x7f8c7c18edc0>",
        "id": 94875142519344,
        "start_time_ns": 1656363168816321130,
        "duration_time_ns": 7957,
        "correlation_id": 0,
        "children": [
            94875145539632
        ],
        "parent": 94875145539216
    },
    {
        "_name": "enum.py(774): __hash__",
        "id": 94875145539632,
        "start_time_ns": 1656363168816322953,
        "duration_time_ns": 5081,
        "correlation_id": 0,
        "children": [
            94875142519760
        ],
        "parent": 94875142519344
    },
    {
        "_name": "<built-in function hash>",
        "id": 94875142519760,
        "start_time_ns": 1656363168816327283,
        "duration_time_ns": 670,
        "correlation_id": 0,
        "children": [],
        "parent": 94875145539632
    },
    {
        "_name": "aten::matmul",
        "id": 94875144444544,
        "start_time_ns": 1656363168816278419,
        "duration_time_ns": 16800,
        "correlation_id": 3141,
        "children": [
            94875144445216
        ],
        "parent": 94875145485184
    },
    {
        "_name": "aten::mm",
        "id": 94875144445216,
        "start_time_ns": 1656363168816278962,
        "duration_time_ns": 15722,
        "correlation_id": 3142,
        "children": [],
        "parent": 94875144444544
    },
    {
        "_name": "aten::matmul",
        "id": 94875144443200,
        "start_time_ns": 1656363168816258456,
        "duration_time_ns": 17607,
        "correlation_id": 3139,
        "children": [
            94875144443872
        ],
        "parent": 94875145485184
    },
    {
        "_name": "aten::mm",
        "id": 94875144443872,
        "start_time_ns": 1656363168816258971,
        "duration_time_ns": 16578,
        "correlation_id": 3140,
        "children": [],
        "parent": 94875144443200
    },
    {
        "_name": "aten::matmul",
        "id": 94875144440080,
        "start_time_ns": 1656363168816228166,
        "duration_time_ns": 27957,
        "correlation_id": 3137,
        "children": [
            94875144442528
        ],
        "parent": 94875145485184
    },
    {
        "_name": "aten::mm",
        "id": 94875144442528,
        "start_time_ns": 1656363168816228620,
        "duration_time_ns": 26961,
        "correlation_id": 3138,
        "children": [],
        "parent": 94875144440080
    },
    {
        "_name": "aten::matmul",
        "id": 94875144438736,
        "start_time_ns": 1656363168816203994,
        "duration_time_ns": 21725,
        "correlation_id": 3135,
        "children": [
            94875144439408
        ],
        "parent": 94875145485184
    },
    {
        "_name": "aten::mm",
        "id": 94875144439408,
        "start_time_ns": 1656363168816204595,
        "duration_time_ns": 20560,
        "correlation_id": 3136,
        "children": [],
        "parent": 94875144438736
    },
    {
        "_name": "aten::matmul",
        "id": 94875144437392,
        "start_time_ns": 1656363168816130349,
        "duration_time_ns": 56214,
        "correlation_id": 3133,
        "children": [
            94875144438064
        ],
        "parent": 94875145485184
    },
    {
        "_name": "aten::mm",
        "id": 94875144438064,
        "start_time_ns": 1656363168816132349,
        "duration_time_ns": 53343,
        "correlation_id": 3134,
        "children": [],
        "parent": 94875144437392
    },
    {
        "_name": "test_profiler.py(1368): garbage_code",
        "id": 94875145537968,
        "start_time_ns": 1656363168762677830,
        "duration_time_ns": 53444135,
        "correlation_id": 0,
        "children": [
            94875115662832,
            94875115663808,
            94875115664784,
            94875115665376,
            94875115666496,
            94875115667968,
            94875115675680,
            94875115677152,
            94875115678624,
            94875115679360,
            94875115680832,
            94875115682304,
            94875115683040,
            94875115684512,
            94875115685984,
            94875115686720,
            94875115688192,
            94875115689664,
            94875115690400,
            94875115691872,
            94875115693344,
            94875115694080,
            94875115695552,
            94875144429440,
            94875144430176,
            94875144431648,
            94875144433120,
            94875144433856,
            94875144435328,
            94875144436800
        ],
        "parent": 94875145485184
    },
    {
        "_name": "aten::copy_",
        "id": 94875144436800,
        "start_time_ns": 1656363168816100255,
        "duration_time_ns": 19556,
        "correlation_id": 3132,
        "children": [],
        "parent": 94875145537968
    },
    {
        "_name": "aten::select",
        "id": 94875144435328,
        "start_time_ns": 1656363168816098116,
        "duration_time_ns": 1371,
        "correlation_id": 3130,
        "children": [
            94875144436240
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875144436240,
        "start_time_ns": 1656363168816098865,
        "duration_time_ns": 217,
        "correlation_id": 3131,
        "children": [],
        "parent": 94875144435328
    },
    {
        "_name": "aten::select",
        "id": 94875144433856,
        "start_time_ns": 1656363168816095449,
        "duration_time_ns": 2017,
        "correlation_id": 3128,
        "children": [
            94875144434768
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875144434768,
        "start_time_ns": 1656363168816096566,
        "duration_time_ns": 393,
        "correlation_id": 3129,
        "children": [],
        "parent": 94875144433856
    },
    {
        "_name": "aten::copy_",
        "id": 94875144433120,
        "start_time_ns": 1656363168816071818,
        "duration_time_ns": 19713,
        "correlation_id": 3127,
        "children": [],
        "parent": 94875145537968
    },
    {
        "_name": "aten::select",
        "id": 94875144431648,
        "start_time_ns": 1656363168816069581,
        "duration_time_ns": 1436,
        "correlation_id": 3125,
        "children": [
            94875144432560
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875144432560,
        "start_time_ns": 1656363168816070353,
        "duration_time_ns": 220,
        "correlation_id": 3126,
        "children": [],
        "parent": 94875144431648
    },
    {
        "_name": "aten::select",
        "id": 94875144430176,
        "start_time_ns": 1656363168816066766,
        "duration_time_ns": 2102,
        "correlation_id": 3123,
        "children": [
            94875144431088
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875144431088,
        "start_time_ns": 1656363168816067949,
        "duration_time_ns": 388,
        "correlation_id": 3124,
        "children": [],
        "parent": 94875144430176
    },
    {
        "_name": "aten::copy_",
        "id": 94875144429440,
        "start_time_ns": 1656363168816043018,
        "duration_time_ns": 19842,
        "correlation_id": 3122,
        "children": [],
        "parent": 94875145537968
    },
    {
        "_name": "aten::select",
        "id": 94875115695552,
        "start_time_ns": 1656363168816040799,
        "duration_time_ns": 1394,
        "correlation_id": 3120,
        "children": [
            94875115696464
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115696464,
        "start_time_ns": 1656363168816041569,
        "duration_time_ns": 222,
        "correlation_id": 3121,
        "children": [],
        "parent": 94875115695552
    },
    {
        "_name": "aten::select",
        "id": 94875115694080,
        "start_time_ns": 1656363168816037496,
        "duration_time_ns": 2629,
        "correlation_id": 3118,
        "children": [
            94875115694992
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115694992,
        "start_time_ns": 1656363168816039133,
        "duration_time_ns": 449,
        "correlation_id": 3119,
        "children": [],
        "parent": 94875115694080
    },
    {
        "_name": "aten::copy_",
        "id": 94875115693344,
        "start_time_ns": 1656363168816013749,
        "duration_time_ns": 19684,
        "correlation_id": 3117,
        "children": [],
        "parent": 94875145537968
    },
    {
        "_name": "aten::select",
        "id": 94875115691872,
        "start_time_ns": 1656363168816011407,
        "duration_time_ns": 1547,
        "correlation_id": 3115,
        "children": [
            94875115692784
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115692784,
        "start_time_ns": 1656363168816012291,
        "duration_time_ns": 243,
        "correlation_id": 3116,
        "children": [],
        "parent": 94875115691872
    },
    {
        "_name": "aten::select",
        "id": 94875115690400,
        "start_time_ns": 1656363168816008574,
        "duration_time_ns": 2145,
        "correlation_id": 3113,
        "children": [
            94875115691312
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115691312,
        "start_time_ns": 1656363168816009816,
        "duration_time_ns": 388,
        "correlation_id": 3114,
        "children": [],
        "parent": 94875115690400
    },
    {
        "_name": "aten::copy_",
        "id": 94875115689664,
        "start_time_ns": 1656363168815985233,
        "duration_time_ns": 19349,
        "correlation_id": 3112,
        "children": [],
        "parent": 94875145537968
    },
    {
        "_name": "aten::select",
        "id": 94875115688192,
        "start_time_ns": 1656363168815982928,
        "duration_time_ns": 1461,
        "correlation_id": 3110,
        "children": [
            94875115689104
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115689104,
        "start_time_ns": 1656363168815983684,
        "duration_time_ns": 294,
        "correlation_id": 3111,
        "children": [],
        "parent": 94875115688192
    },
    {
        "_name": "aten::select",
        "id": 94875115686720,
        "start_time_ns": 1656363168815980155,
        "duration_time_ns": 2131,
        "correlation_id": 3108,
        "children": [
            94875115687632
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115687632,
        "start_time_ns": 1656363168815981345,
        "duration_time_ns": 407,
        "correlation_id": 3109,
        "children": [],
        "parent": 94875115686720
    },
    {
        "_name": "aten::copy_",
        "id": 94875115685984,
        "start_time_ns": 1656363168815956492,
        "duration_time_ns": 19456,
        "correlation_id": 3107,
        "children": [],
        "parent": 94875145537968
    },
    {
        "_name": "aten::select",
        "id": 94875115684512,
        "start_time_ns": 1656363168815954189,
        "duration_time_ns": 1483,
        "correlation_id": 3105,
        "children": [
            94875115685424
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115685424,
        "start_time_ns": 1656363168815955044,
        "duration_time_ns": 244,
        "correlation_id": 3106,
        "children": [],
        "parent": 94875115684512
    },
    {
        "_name": "aten::select",
        "id": 94875115683040,
        "start_time_ns": 1656363168815950182,
        "duration_time_ns": 3038,
        "correlation_id": 3103,
        "children": [
            94875115683952
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115683952,
        "start_time_ns": 1656363168815952213,
        "duration_time_ns": 449,
        "correlation_id": 3104,
        "children": [],
        "parent": 94875115683040
    },
    {
        "_name": "aten::copy_",
        "id": 94875115682304,
        "start_time_ns": 1656363168815926164,
        "duration_time_ns": 20024,
        "correlation_id": 3102,
        "children": [],
        "parent": 94875145537968
    },
    {
        "_name": "aten::select",
        "id": 94875115680832,
        "start_time_ns": 1656363168815923748,
        "duration_time_ns": 1588,
        "correlation_id": 3100,
        "children": [
            94875115681744
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115681744,
        "start_time_ns": 1656363168815924664,
        "duration_time_ns": 238,
        "correlation_id": 3101,
        "children": [],
        "parent": 94875115680832
    },
    {
        "_name": "aten::select",
        "id": 94875115679360,
        "start_time_ns": 1656363168815920928,
        "duration_time_ns": 2174,
        "correlation_id": 3098,
        "children": [
            94875115680272
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115680272,
        "start_time_ns": 1656363168815922170,
        "duration_time_ns": 407,
        "correlation_id": 3099,
        "children": [],
        "parent": 94875115679360
    },
    {
        "_name": "aten::copy_",
        "id": 94875115678624,
        "start_time_ns": 1656363168815896438,
        "duration_time_ns": 20352,
        "correlation_id": 3097,
        "children": [],
        "parent": 94875145537968
    },
    {
        "_name": "aten::select",
        "id": 94875115677152,
        "start_time_ns": 1656363168815893997,
        "duration_time_ns": 1560,
        "correlation_id": 3095,
        "children": [
            94875115678064
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115678064,
        "start_time_ns": 1656363168815894821,
        "duration_time_ns": 259,
        "correlation_id": 3096,
        "children": [],
        "parent": 94875115677152
    },
    {
        "_name": "aten::select",
        "id": 94875115675680,
        "start_time_ns": 1656363168815890536,
        "duration_time_ns": 2715,
        "correlation_id": 3093,
        "children": [
            94875115676592
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115676592,
        "start_time_ns": 1656363168815892242,
        "duration_time_ns": 430,
        "correlation_id": 3094,
        "children": [],
        "parent": 94875115675680
    },
    {
        "_name": "aten::copy_",
        "id": 94875115667968,
        "start_time_ns": 1656363168815856554,
        "duration_time_ns": 29186,
        "correlation_id": 3092,
        "children": [],
        "parent": 94875145537968
    },
    {
        "_name": "aten::select",
        "id": 94875115666496,
        "start_time_ns": 1656363168815852790,
        "duration_time_ns": 2196,
        "correlation_id": 3090,
        "children": [
            94875115667408
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115667408,
        "start_time_ns": 1656363168815854098,
        "duration_time_ns": 355,
        "correlation_id": 3091,
        "children": [],
        "parent": 94875115666496
    },
    {
        "_name": "aten::select",
        "id": 94875115665376,
        "start_time_ns": 1656363168815847567,
        "duration_time_ns": 4419,
        "correlation_id": 3088,
        "children": [
            94875115665936
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115665936,
        "start_time_ns": 1656363168815849930,
        "duration_time_ns": 1088,
        "correlation_id": 3089,
        "children": [],
        "parent": 94875115665376
    },
    {
        "_name": "aten::copy_",
        "id": 94875115664784,
        "start_time_ns": 1656363168772960015,
        "duration_time_ns": 42868971,
        "correlation_id": 3087,
        "children": [],
        "parent": 94875145537968
    },
    {
        "_name": "aten::select",
        "id": 94875115663808,
        "start_time_ns": 1656363168772291697,
        "duration_time_ns": 664130,
        "correlation_id": 3085,
        "children": [
            94875115664368
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115664368,
        "start_time_ns": 1656363168772953425,
        "duration_time_ns": 974,
        "correlation_id": 3086,
        "children": [],
        "parent": 94875115663808
    },
    {
        "_name": "aten::select",
        "id": 94875115662832,
        "start_time_ns": 1656363168764614964,
        "duration_time_ns": 7673288,
        "correlation_id": 3083,
        "children": [
            94875115663392
        ],
        "parent": 94875145537968
    },
    {
        "_name": "aten::as_strided",
        "id": 94875115663392,
        "start_time_ns": 1656363168770373721,
        "duration_time_ns": 3845,
        "correlation_id": 3084,
        "children": [],
        "parent": 94875115662832
    },
    {
        "_name": "aten::matmul",
        "id": 94875115652976,
        "start_time_ns": 1656363168762656764,
        "duration_time_ns": 17095,
        "correlation_id": 3081,
        "children": [
            94875115653648
        ],
        "parent": 94875145485184
    },
    {
        "_name": "aten::mm",
        "id": 94875115653648,
        "start_time_ns": 1656363168762657214,
        "duration_time_ns": 16088,
        "correlation_id": 3082,
        "children": [],
        "parent": 94875115652976
    },
    {
        "_name": "aten::matmul",
        "id": 94873884074288,
        "start_time_ns": 1656363168762636786,
        "duration_time_ns": 17470,
        "correlation_id": 3079,
        "children": [
            94875115652304
        ],
        "parent": 94875145485184
    },
    {
        "_name": "aten::mm",
        "id": 94875115652304,
        "start_time_ns": 1656363168762637261,
        "duration_time_ns": 16372,
        "correlation_id": 3080,
        "children": [],
        "parent": 94873884074288
    },
    {
        "_name": "aten::matmul",
        "id": 94873869202528,
        "start_time_ns": 1656363168762607057,
        "duration_time_ns": 27149,
        "correlation_id": 3077,
        "children": [
            94873884078048
        ],
        "parent": 94875145485184
    },
    {
        "_name": "aten::mm",
        "id": 94873884078048,
        "start_time_ns": 1656363168762607522,
        "duration_time_ns": 26074,
        "correlation_id": 3078,
        "children": [],
        "parent": 94873869202528
    },
    {
        "_name": "aten::matmul",
        "id": 94873875387024,
        "start_time_ns": 1656363168762572460,
        "duration_time_ns": 31587,
        "correlation_id": 3075,
        "children": [
            94873872544160
        ],
        "parent": 94875145485184
    },
    {
        "_name": "aten::mm",
        "id": 94873872544160,
        "start_time_ns": 1656363168762574240,
        "duration_time_ns": 28970,
        "correlation_id": 3076,
        "children": [],
        "parent": 94873875387024
    },
    {
        "_name": "aten::matmul",
        "id": 94873905411152,
        "start_time_ns": 1656363168761401368,
        "duration_time_ns": 1153619,
        "correlation_id": 3073,
        "children": [
            94873876190080
        ],
        "parent": 94875145485184
    },
    {
        "_name": "aten::mm",
        "id": 94873876190080,
        "start_time_ns": 1656363168761406718,
        "duration_time_ns": 1145188,
        "correlation_id": 3074,
        "children": [],
        "parent": 94873905411152
    },
    {
        "_name": "torch/profiler/profiler.py(470): __enter__",
        "id": 94875145485600,
        "start_time_ns": 1656363168760431422,
        "duration_time_ns": 87171,
        "correlation_id": 0,
        "children": [
            94875145486016
        ],
        "parent": 94875145485184
    },
    {
        "_name": "torch/profiler/profiler.py(477): start",
        "id": 94875145486016,
        "start_time_ns": 1656363168760434072,
        "duration_time_ns": 84390,
        "correlation_id": 0,
        "children": [
            94875145486432
        ],
        "parent": 94875145485600
    },
    {
        "_name": "torch/profiler/profiler.py(513): _transit_action",
        "id": 94875145486432,
        "start_time_ns": 1656363168760434457,
        "duration_time_ns": 83558,
        "correlation_id": 0,
        "children": [
            94875145486848
        ],
        "parent": 94875145486016
    },
    {
        "_name": "torch/profiler/profiler.py(108): start_trace",
        "id": 94875145486848,
        "start_time_ns": 1656363168760435492,
        "duration_time_ns": 82129,
        "correlation_id": 0,
        "children": [
            94875145536304,
            94875142518480,
            94875145536720
        ],
        "parent": 94875145486432
    },
    {
        "_name": "torch/profiler/profiler.py(187): _get_distributed_info",
        "id": 94875145536720,
        "start_time_ns": 1656363168760505094,
        "duration_time_ns": 12311,
        "correlation_id": 0,
        "children": [
            94875145537136,
            94875145537552
        ],
        "parent": 94875145486848
    },
    {
        "_name": "torch/distributed/distributed_c10d.py(415): is_initialized",
        "id": 94875145537552,
        "start_time_ns": 1656363168760515507,
        "duration_time_ns": 1805,
        "correlation_id": 0,
        "children": [],
        "parent": 94875145536720
    },
    {
        "_name": "torch/distributed/__init__.py(8): is_available",
        "id": 94875145537136,
        "start_time_ns": 1656363168760510479,
        "duration_time_ns": 3242,
        "correlation_id": 0,
        "children": [
            94875142518928
        ],
        "parent": 94875145536720
    },
    {
        "_name": "<built-in function hasattr>",
        "id": 94875142518928,
        "start_time_ns": 1656363168760513054,
        "duration_time_ns": 571,
        "correlation_id": 0,
        "children": [],
        "parent": 94875145537136
    },
    {
        "_name": "<built-in method kineto_available of PyCapsule object at 0x7f8c84b00120>",
        "id": 94875142518480,
        "start_time_ns": 1656363168760503357,
        "duration_time_ns": 605,
        "correlation_id": 0,
        "children": [],
        "parent": 94875145486848
    },
    {
        "_name": "torch/autograd/profiler.py(205): _start_trace",
        "id": 94875145536304,
        "start_time_ns": 1656363168760436852,
        "duration_time_ns": 53664,
        "correlation_id": 0,
        "children": [],
        "parent": 94875145486848
    }
]'''
