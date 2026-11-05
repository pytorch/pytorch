# Owner(s): ["oncall: profiler"]

from unittest.mock import patch

import torch
from torch.profiler import profile, ProfilerAction, ProfilerActivity
from torch.testing._internal.common_profiler import forbid_profiler_capture
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class _RecordingProfile(profile):
    def __init__(
        self,
        *,
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        **kwargs,
    ):
        self.actions: list[str] = []
        # The default CUDA activity exercises the use_device is not None guard
        # in step(), without requiring hardware.
        super().__init__(activities=activities, **kwargs)

    def prepare_trace(self):
        self.actions.append("prepare")

    def start_trace(self):
        self.actions.append("start")

    def stop_trace(self):
        self.actions.append("stop")


@instantiate_parametrized_tests
class TestProfilerSchedule(TestCase):
    hw_classification = HardwareClassification.GENERIC
    PATCH_TARGET = "torch.autograd._is_kineto_stopped"

    def setUp(self):
        super().setUp()
        # Fail if a test accidentally uses a real capture instead of the fake.
        self.enterContext(forbid_profiler_capture())
        self.enterContext(patch(self.PATCH_TARGET, return_value=False))
        self.enterContext(patch("torch.profiler.profiler.prof.KinetoStepTracker"))
        self.enterContext(patch("torch.profiler.profiler.prof.record_function"))
        self.enterContext(
            patch.object(
                torch.profiler.ExecutionTraceObserver,
                "build_execution_trace_obs_from_env",
                return_value=None,
            )
        )

    def _make_profiler(self, **schedule_kwargs):
        return _RecordingProfile(schedule=torch.profiler.schedule(**schedule_kwargs))

    def test_enters_device_stopped_from_warmup(self):
        p = self._make_profiler(wait=0, warmup=2, active=2)
        with patch(self.PATCH_TARGET, return_value=True):
            p.start()
            # step 0 -> 1: WARMUP -> WARMUP per schedule, but _is_kineto_stopped
            # returns True so we override to DEVICE_STOPPED.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            p.stop()
        self.assertEqual(p.actions, ["prepare", "start", "stop"])

    def test_enters_device_stopped_from_record(self):
        p = self._make_profiler(wait=0, warmup=1, active=3)
        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
            # step 0 -> 1: WARMUP -> RECORD (normal)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)
            self.assertEqual(p.actions, ["prepare", "start"])

        with patch(self.PATCH_TARGET, return_value=True):
            # step 1 -> 2: RECORD -> RECORD per schedule, but _is_kineto_stopped
            # -> True overrides to DEVICE_STOPPED.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            p.stop()
        self.assertEqual(p.actions, ["prepare", "start", "stop"])

    def test_does_not_enter_device_stopped_from_record_and_save(self):
        """When prev_action is RECORD_AND_SAVE, the natural action_map
        transition (e.g. R&S -> WARMUP) already fires stop_trace +
        _trace_ready + prepare_trace, which resets Kineto. We do NOT
        override to DEVICE_STOPPED -- doing so would skip the reset and
        waste the next cycle in DEVICE_STOPPED. If Kineto recovers,
        profiling continues normally; if not, the entry guard fires on
        the next step (prev=WARMUP or RECORD)."""
        callback_count = [0]

        def handler(prof):
            callback_count[0] += 1

        p = _RecordingProfile(
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=2),
            on_trace_ready=handler,
        )

        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD_AND_SAVE)

        with patch(self.PATCH_TARGET, return_value=True):
            # step 2 -> 3: prev=R&S, schedule=WARMUP. Despite kineto_stopped,
            # we do NOT override; the natural (R&S, WARMUP) transition fires
            # stop_trace + _trace_ready + prepare_trace.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            self.assertEqual(p.actions, ["prepare", "start", "stop", "prepare"])
            self.assertEqual(
                callback_count[0],
                1,
                "(R&S, WARMUP) natural transition must fire on_trace_ready",
            )

        # Simulate Kineto recovering after prepare_trace: cycle 2 records
        # normally rather than being wasted in DEVICE_STOPPED.
        with patch(self.PATCH_TARGET, return_value=False):
            p.step()  # step 3 -> 4: WARMUP -> RECORD
            self.assertEqual(p.current_action, ProfilerAction.RECORD)
            p.step()  # step 4 -> 5: RECORD -> RECORD_AND_SAVE
            self.assertEqual(p.current_action, ProfilerAction.RECORD_AND_SAVE)
            p.stop()
            # cycle 2 R&S -> stop fires on_trace_ready a second time
            self.assertEqual(callback_count[0], 2)

    def test_device_stopped_persists_through_cycle(self):
        """Once in DEVICE_STOPPED, the profiler stays there until the schedule
        moves to WARMUP (new cycle) or NONE (end)."""
        p = self._make_profiler(wait=0, warmup=1, active=4, repeat=2)
        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)

        with patch(self.PATCH_TARGET, return_value=True):
            # step 1 -> 2: RECORD -> DEVICE_STOPPED
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            # step 2 -> 3: schedule says RECORD, stays DEVICE_STOPPED
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            # step 3 -> 4: schedule says RECORD_AND_SAVE, stays DEVICE_STOPPED
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            self.assertEqual(p.actions, ["prepare", "start", "stop"])
            # step 4 -> 5: schedule rolls over to WARMUP (new cycle), exits
            # DEVICE_STOPPED.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            self.assertEqual(p.actions, ["prepare", "start", "stop", "prepare"])
            p.stop()

    def test_device_stopped_recovers_without_warmup(self):
        """With warmup=0 (and wait=0), the schedule never produces WARMUP or
        NONE between cycles. DEVICE_STOPPED must still exit at cycle
        boundaries (RECORD_AND_SAVE -> next), otherwise the profiler would
        be stuck in DEVICE_STOPPED forever."""
        # The normal schedule for wait=0, warmup=0, active=3, repeat=2:
        # step 0..1: RECORD
        # step 2:    RECORD_AND_SAVE
        # step 3..4: RECORD
        # step 5:    RECORD_AND_SAVE
        # step 6+:   NONE
        p = self._make_profiler(wait=0, warmup=0, active=3, repeat=2)

        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)

        with patch(self.PATCH_TARGET, return_value=True):
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            # step 2 -> 3: prev raw schedule output was RECORD_AND_SAVE, so
            # this step is a cycle boundary. Exit DEVICE_STOPPED into
            # current_action (RECORD)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)
            self.assertEqual(
                p.actions, ["prepare", "start", "stop", "prepare", "start"]
            )
            p.stop()

    def test_device_stopped_exits_via_wait_phase(self):
        """With wait > 0, the NONE phase between cycles is the natural
        exit path out of DEVICE_STOPPED. After the wait, profiling resumes
        normally through WARMUP and RECORD."""
        # The normal schedule for wait=2, warmup=1, active=2, repeat=2 (cycle length 5):
        # step 0..1: NONE
        # step 2:    WARMUP
        # step 3:    RECORD
        # step 4:    R&S
        # step 5..6: NONE
        # step 7:    WARMUP
        # step 8:    RECORD
        # step 9:    R&S
        # step 10+:  NONE
        p = self._make_profiler(wait=2, warmup=1, active=2, repeat=2)

        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.NONE)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)

        with patch(self.PATCH_TARGET, return_value=True):
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)

        with patch(self.PATCH_TARGET, return_value=False):
            # step 4 -> 5: schedule rolls into cycle 2's wait phase. NONE is
            # in the exit set, so DEVICE_STOPPED exits.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.NONE)
            # step 5 -> 6: still in wait phase
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.NONE)
            self.assertEqual(p.actions, ["prepare", "start", "stop"])
            # step 6 -> 7: WARMUP (cycle 2 begins)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            self.assertEqual(p.actions, ["prepare", "start", "stop", "prepare"])
            # step 7 -> 8: RECORD (cycle 2 active begins)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)
            self.assertEqual(
                p.actions, ["prepare", "start", "stop", "prepare", "start"]
            )
            p.stop()

    def test_device_stopped_persists_through_multistep_warmup(self):
        """With warmup >= 2, consecutive WARMUP steps are mid-cycle, not
        cycle boundaries. DEVICE_STOPPED must persist through the entire
        warmup phase rather than oscillating WARMUP -> DEVICE_STOPPED ->
        WARMUP -> DEVICE_STOPPED on every step."""
        # The normal schedule for wait=0, warmup=3, active=2, repeat=1 (cycle length 5):
        # step 0..2: WARMUP
        # step 3:    RECORD
        # step 4:    R&S
        # step 5+:   NONE
        p = self._make_profiler(wait=0, warmup=3, active=2, repeat=1)

        with patch(self.PATCH_TARGET, return_value=True):
            p.start()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            # step 1 -> 2: schedule still WARMUP (same warmup phase, NOT a
            # cycle boundary). Must stay in DEVICE_STOPPED.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            # step 2 -> 3: schedule moves to RECORD. Still mid-cycle
            # (prev_schedule_action is WARMUP, not RECORD_AND_SAVE). Stay.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            # step 3 -> 4: schedule moves to R&S. Still mid-cycle. Stay.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            # step 4 -> 5: prev_schedule_action == R&S marks the cycle
            # boundary (here also end-of-profiling since repeat=1). Exit
            # DEVICE_STOPPED into NONE.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.NONE)
            p.stop()
        self.assertEqual(p.actions, ["prepare", "start", "stop"])

    def test_warmup_to_device_stopped_fires_trace_ready(self):
        """When DEVICE_STOPPED is entered from WARMUP (e.g. active=1
        schedule where every R&S step gets converted to DS), the
        (WARMUP, DEVICE_STOPPED) transition must fire on_trace_ready so
        the user's callback is invoked exactly once per cycle. Without
        this, active=1 schedules with persistent kineto_stopped would
        silently drop all callbacks; with double-firing we'd produce
        spurious callbacks."""
        callback_count = [0]

        def handler(prof):
            callback_count[0] += 1

        # Schedule active=1, warmup=1, repeat=3 cycles W, R&S; with kineto_stopped
        # on every step the entry guard converts each cycle's R&S to DS via
        # (W, DS), and DS exits at cycle boundary via (DS, W).
        p = _RecordingProfile(
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=3),
            on_trace_ready=handler,
        )

        with patch(self.PATCH_TARGET, return_value=False):
            p.start()

        with patch(self.PATCH_TARGET, return_value=True):
            # Cycle 1: step 0 -> 1: (W, DS) fires _trace_ready. callback=1.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            self.assertEqual(callback_count[0], 1)
            # step 1 -> 2: (DS, W) fires prepare_trace, no callback. callback=1.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            self.assertEqual(callback_count[0], 1)
            # Cycle 2: step 2 -> 3: (W, DS) fires _trace_ready. callback=2.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            self.assertEqual(callback_count[0], 2)
            p.stop()

    def test_start_after_stop_in_device_stopped(self):
        """Stopping in DEVICE_STOPPED must restore the scheduled action so a later
        start() issues prepare/start again."""
        callback_count = [0]

        def handler(prof):
            callback_count[0] += 1

        p = _RecordingProfile(
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=1),
            on_trace_ready=handler,
        )

        # Run 1: enter DEVICE_STOPPED, stop while in it.
        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
        with patch(self.PATCH_TARGET, return_value=True):
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            p.stop()
        # (W, DS) fired _trace_ready during run 1.
        self.assertEqual(callback_count[0], 1)

        # stop() restored R&S from the raw schedule at step 1, so the next
        # start() must issue prepare_trace + start_trace again.
        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
            self.assertEqual(p.current_action, ProfilerAction.RECORD_AND_SAVE)
            self.assertEqual(
                p.actions, ["prepare", "start", "stop", "prepare", "start"]
            )
            p.stop()
        # (R&S, None) fires _trace_ready -> callback_count increments.
        self.assertEqual(
            callback_count[0],
            2,
            "second run must fire the trace-ready callback",
        )

    def test_user_schedule_returning_device_stopped_raises_at_init(self):
        """ProfilerAction.DEVICE_STOPPED is internal; user-provided
        schedules must not return it. __init__ catches schedules that
        return it at step 0 -- without this check, start() would silently
        no-op the (NONE, DS) transition and leave the profiler dead."""
        with self.assertRaisesRegex(ValueError, "DEVICE_STOPPED is set internally"):
            _RecordingProfile(schedule=lambda step: ProfilerAction.DEVICE_STOPPED)

    def test_user_schedule_returning_device_stopped_raises_at_step(self):
        """If the user schedule returns DEVICE_STOPPED on a later step,
        step() raises ValueError and leaves the profiler state untouched
        so the user can stop() cleanly after catching the error."""

        def bad_schedule(step):
            if step == 0:
                return ProfilerAction.WARMUP
            return ProfilerAction.DEVICE_STOPPED

        p = _RecordingProfile(schedule=bad_schedule)
        p.start()
        step_num_before = p.step_num
        current_action_before = p.current_action
        with self.assertRaisesRegex(ValueError, "DEVICE_STOPPED is set internally"):
            p.step()
        # Profiler state must be unchanged so stop() can finalize cleanly.
        self.assertEqual(p.step_num, step_num_before)
        self.assertEqual(p.current_action, current_action_before)
        # step() has already exited the step annotation before raising.
        self.assertIsNone(p.step_rec_fn)
        p.stop()

    def test_device_stopped_recovers_across_infinite_cycles(self):
        """With repeat=0 (infinite cycles), DEVICE_STOPPED entry and
        recovery must work across multiple cycles, not just one."""
        # Normal schedule for wait=0, warmup=1, active=2, repeat=0 (cycle length 3, infinite):
        # step 0: WARMUP
        # step 1: RECORD
        # step 2: RECORD_AND_SAVE
        # step 3: WARMUP
        # step 4: RECORD
        # step 5: RECORD_AND_SAVE
        # step 6: WARMUP
        # ...
        p = self._make_profiler(wait=0, warmup=1, active=2, repeat=0)

        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)

        # Cycle 1: enter DS at the R&S step.
        with patch(self.PATCH_TARGET, return_value=True):
            p.step()  # step 2: prev=R, schedule=R&S -> DS
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)

        # Cycle 1 -> Cycle 2: exit DS at the R&S boundary.
        with patch(self.PATCH_TARGET, return_value=False):
            p.step()  # step 3: prev=DS, prev_schedule=R&S -> WARMUP
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.step()  # step 4: WARMUP -> RECORD
            self.assertEqual(p.current_action, ProfilerAction.RECORD)

        # Cycle 2: enter DS again.
        with patch(self.PATCH_TARGET, return_value=True):
            p.step()  # step 5: prev=R, schedule=R&S -> DS
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)

        # Cycle 2 -> Cycle 3: exit again. Confirms recovery is not a
        # one-shot -- it works across arbitrary cycles.
        with patch(self.PATCH_TARGET, return_value=False):
            p.step()  # step 6: prev=DS, prev_schedule=R&S -> WARMUP
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.stop()

    def test_skips_device_stopped_when_prev_is_none(self):
        p = self._make_profiler(wait=1, warmup=1, active=1)
        with patch(self.PATCH_TARGET, return_value=True):
            p.start()
            self.assertEqual(p.current_action, ProfilerAction.NONE)
            # NONE -> WARMUP satisfies every entry condition except prev_action.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.stop()

    @parametrize(
        "prev",
        [ProfilerAction.WARMUP, ProfilerAction.RECORD],
        name_fn=lambda action: action.name.lower(),
    )
    def test_skips_device_stopped_when_already_stopping(self, prev):
        p = _RecordingProfile(
            schedule=lambda step: prev if step == 0 else ProfilerAction.NONE
        )
        p.start()
        self.assertEqual(p.current_action, prev)
        with patch(self.PATCH_TARGET, return_value=True):
            # The previous action permits entry; only current_action blocks it.
            with self.assertWarnsRegex(
                UserWarning, f"Incorrect schedule: {prev.name} followed by NONE"
            ):
                p.step()
            self.assertEqual(p.current_action, ProfilerAction.NONE)
            p.stop()
        self.assertEqual(p.actions, ["prepare", "start", "stop"])

    @parametrize(
        "prev",
        [ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE],
        name_fn=lambda action: action.name.lower(),
    )
    def test_unreachable_transition_raises(self, prev):
        """Transitions into DEVICE_STOPPED that step() / start() / __init__
        all exclude should raise if invoked directly via _transit_action.

        - prev=NONE: step()'s entry guard excludes it, and start() /
          __init__ reject DEVICE_STOPPED from user schedules, so
          current_action is never DEVICE_STOPPED at _transit_action(NONE, ...).
        - prev=RECORD_AND_SAVE: entry guard excludes it because the
          natural (R&S, *) transitions handle cycle-boundary cleanup.
        """
        p = self._make_profiler(wait=0, warmup=1, active=1)
        with self.assertRaisesRegex(
            RuntimeError,
            f"Profiler internal error: {prev.name} -> DEVICE_STOPPED should be unreachable",
        ):
            p._transit_action(prev, ProfilerAction.DEVICE_STOPPED)

    def test_cpu_only_profiler_ignores_stale_is_kineto_stopped(self):
        """A CPU-only profiler should not enter DEVICE_STOPPED even if
        _is_kineto_stopped returns True (e.g. stale flag from a previous
        device profiler)."""
        p = _RecordingProfile(
            activities=[ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=2),
        )
        with patch("torch.autograd._is_kineto_stopped", return_value=True):
            p.start()
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD_AND_SAVE)
            p.stop()

    def test_default_skip_first_wait(self):
        test_schedule = torch.profiler.schedule(
            skip_first=3, wait=2, warmup=1, active=4, repeat=2
        )
        test_schedule_expected_outputs = [
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
        ]
        for step in range(len(test_schedule_expected_outputs)):
            self.assertEqual(test_schedule(step), test_schedule_expected_outputs[step])

    def test_skip_first_wait(self):
        test_schedule = torch.profiler.schedule(
            skip_first=3, wait=5, warmup=1, active=2, repeat=2, skip_first_wait=1
        )
        test_schedule_expected_outputs = [
            # repeat No. 1 begin
            # skip first 3
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            # warmup 1
            ProfilerAction.WARMUP,
            # active 1 begin
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            # active 1 end
            # repeat No. 1 end
            # ---
            # repeat No. 2 begin
            # wait 5
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            # warmup 1
            ProfilerAction.WARMUP,
            # active 2 begin
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            # active 2 end
            # repeat No. 2 end
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
        ]
        for step in range(len(test_schedule_expected_outputs)):
            self.assertEqual(test_schedule(step), test_schedule_expected_outputs[step])


if __name__ == "__main__":
    run_tests()
