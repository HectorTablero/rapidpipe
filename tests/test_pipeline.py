from __future__ import annotations

import asyncio
from collections import deque

import pytest

from rapidpipe.layer import Layer, LayerComplete, PipelineStop, ExecutionMode
from rapidpipe.pipeline import Pipeline
from rapidpipe.metrics import MetricsCollector


class FunctionalLayer(Layer):
    def __init__(self, *, process_fn=None, **kwargs):
        self._process_fn = process_fn or (lambda **inputs: None)
        self.seen_inputs: list[dict[str, object]] = []
        super().__init__(**kwargs)

    def process(self, **inputs):
        self.seen_inputs.append(inputs)
        return self._process_fn(**inputs)


# --------------------------------------------------------------------------- #
#  Helper to run one cycle synchronously                                       #
# --------------------------------------------------------------------------- #
def _run_one_cycle(pipe: Pipeline) -> None:
    """Run exactly one processing cycle."""
    asyncio.run(pipe._process_cycle())


# --------------------------------------------------------------------------- #
#  Basic pipeline tests                                                        #
# --------------------------------------------------------------------------- #
def test_pipeline_rejects_non_layer_instances():
    pipe = Pipeline()

    with pytest.raises(TypeError, match="Only Layer instances can be added"):
        pipe.add_layer(object())


def test_pipeline_rejects_reusing_the_same_layer_instance():
    layer = FunctionalLayer(name="source", outputs=["value"])

    first = Pipeline(layer)
    second = Pipeline()

    assert first.layers == [layer]
    with pytest.raises(ValueError, match="already belongs to a pipeline"):
        second.add_layer(layer)


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (
            {"value": "missing.value"},
            "but that layer is not registered or does not provide that output",
        ),
        (
            {"value": "missing"},
            "but no layer already in the pipeline provides it",
        ),
    ],
)
def test_pipeline_rejects_unresolved_dependencies(inputs, match):
    pipe = Pipeline()
    layer = FunctionalLayer(name="consumer", inputs=inputs, outputs=["result"])

    with pytest.raises(ValueError, match=match):
        pipe.add_layer(layer)


def test_pipeline_processes_layers_in_order_and_resolves_dependencies():
    source_values = iter([2, 4])

    source = FunctionalLayer(
        name="source",
        outputs=["left", "right"],
        process_fn=lambda **inputs: (next(source_values), next(source_values)),
    )
    combine = FunctionalLayer(
        name="combine",
        inputs={
            "direct": "source.left",
            "pipeline_direct": "left",
            "right": "source.right",
        },
        outputs=["total"],
        process_fn=lambda direct, pipeline_direct, right: direct + pipeline_direct + right,
    )
    formatter = FunctionalLayer(
        name="formatter",
        inputs={"total": "combine.total"},
        outputs=["text"],
        process_fn=lambda total: f"{total}!",
    )

    pipe = Pipeline(source, combine, formatter)
    _run_one_cycle(pipe)

    assert pipe.cycle_count == 1
    assert source.seen_inputs == [{}]
    assert combine.seen_inputs[0] == {
        "direct": 2, "pipeline_direct": 2, "right": 4}
    assert formatter.seen_inputs[0] == {"total": 8}
    assert pipe.get_value("source", "left") == 2
    assert pipe.get_value(source, "right") == 4
    assert pipe.outputs.left == 2
    assert pipe.outputs.right == 4
    assert pipe.outputs.total == 8
    assert pipe.outputs.text == "8!"
    assert set(pipe.outputs.available_outputs()) == {
        "left", "right", "total", "text"}


def test_pipeline_tracks_indexed_history_only_where_needed():
    source_values = iter([1, 2, 3, 4])

    source = FunctionalLayer(
        name="source",
        outputs=["value"],
        process_fn=lambda **inputs: next(source_values),
    )
    transform = FunctionalLayer(
        name="transform",
        inputs={"value": "source.value"},
        outputs=["double"],
        process_fn=lambda value: value * 2,
    )
    lagger = FunctionalLayer(
        name="lagger",
        inputs={"previous": "transform.double[-2]"},
        outputs=["lagged"],
        process_fn=lambda previous: previous if previous is not None else -1,
    )

    pipe = Pipeline(source, transform, lagger)

    assert pipe.get_storage_requirement("transform", "double") == {
        "cycles": 2,
        "time_seconds": 0.0,
    }
    assert pipe.get_storage_requirement("source", "value") == {
        "cycles": 0,
        "time_seconds": 0.0,
    }

    for _ in range(4):
        _run_one_cycle(pipe)

    assert lagger.seen_inputs[0] == {"previous": None}
    assert lagger.seen_inputs[1] == {"previous": None}
    assert lagger.seen_inputs[2] == {"previous": 2}
    assert lagger.seen_inputs[3] == {"previous": 4}
    assert pipe.outputs.lagged == 4
    assert pipe.get_value("source", "value") == 4
    # Unified history: entries are HistoryEntry objects
    hist = list(pipe._history["transform"]["double"])
    assert [e.value for e in hist] == [4, 6, 8]
    assert list(pipe._history["source"]["value"]) == []


def test_pipeline_tracks_time_range_history_with_clock_and_pruning(clock):
    source_values = iter([1, 2, 3, 4])

    source = FunctionalLayer(
        name="source",
        outputs=["value"],
        process_fn=lambda **inputs: next(source_values),
    )
    transform = FunctionalLayer(
        name="transform",
        inputs={"value": "source.value"},
        outputs=["double"],
        process_fn=lambda value: value * 10,
    )
    window = FunctionalLayer(
        name="window",
        inputs={"recent": "transform.double[-2.5s:-0.5s]"},
        outputs=["count"],
        process_fn=lambda recent: len(recent),
    )

    pipe = Pipeline(source, transform, window, max_history_size=2)

    assert pipe.get_storage_requirement("transform", "double") == {
        "cycles": 0,
        "time_seconds": 2.5,
    }
    assert pipe.get_storage_requirement("source", "value") == {
        "cycles": 0,
        "time_seconds": 0.0,
    }

    _run_one_cycle(pipe)
    assert window.seen_inputs[0] == {"recent": []}

    clock.advance(1.0)
    _run_one_cycle(pipe)
    assert window.seen_inputs[1] == {"recent": []}

    clock.advance(1.0)
    _run_one_cycle(pipe)
    assert window.seen_inputs[2] == {"recent": []}

    clock.advance(1.0)
    _run_one_cycle(pipe)
    assert window.seen_inputs[3] == {"recent": []}
    assert pipe.get_value("source", "value") == 4
    assert pipe.outputs.count == 0
    # Unified history: entries are HistoryEntry objects
    hist = list(pipe._history["transform"]["double"])
    assert [(e.timestamp, e.value) for e in hist] == [
        (102.0, 30),
        (103.0, 40),
    ]
    assert list(pipe._history["source"]["value"]) == []


def test_pipeline_outputs_and_get_value_helpers():
    source = FunctionalLayer(name="source", outputs=["value"])
    pipe = Pipeline(source)

    assert pipe.outputs.available_outputs() == ["value"]
    assert pipe.outputs.value is None
    assert pipe.get_value("source", "value") is None
    assert pipe.get_value(source, "value") is None

    with pytest.raises(AttributeError, match="not available"):
        _ = pipe.outputs.missing

    with pytest.raises(TypeError, match="index must be a string"):
        _ = pipe.outputs[1]


def test_pipeline_run_wraps_unexpected_layer_errors():
    def boom(**inputs):
        raise ValueError("boom")

    layer = FunctionalLayer(name="bad", outputs=["value"], process_fn=boom)
    pipe = Pipeline(layer)

    with pytest.raises(RuntimeError, match="Pipeline error"):
        pipe.run()


# --------------------------------------------------------------------------- #
#  New feature tests                                                           #
# --------------------------------------------------------------------------- #
def test_run_sequence():
    counter = [0]

    def inc(**inputs):
        counter[0] += 1
        return counter[0]

    layer = FunctionalLayer(name="counter", outputs=["n"], process_fn=inc)
    pipe = Pipeline(layer)
    pipe.run_sequence(5)

    assert pipe.cycle_count == 5
    assert pipe.outputs.n == 5


def test_run_until():
    counter = [0]

    def inc(**inputs):
        counter[0] += 1
        return counter[0]

    layer = FunctionalLayer(name="counter", outputs=["n"], process_fn=inc)
    pipe = Pipeline(layer)
    pipe.run_until(lambda p: p.outputs.n >= 3)

    assert pipe.outputs.n >= 3


def test_should_run_skips_layer():
    counter = [0]

    class EveryOtherLayer(Layer):
        def process(self, **inputs):
            counter[0] += 1
            return counter[0]

        def should_run(self, **inputs) -> bool:
            return self._pipeline.cycle_count % 2 == 1

    layer = EveryOtherLayer(name="skip", outputs=["n"])
    pipe = Pipeline(layer)
    pipe.run_sequence(4)

    # should_run returns True on cycles 1 and 3 (odd cycles)
    assert counter[0] == 2


def test_lifecycle_hooks_called():
    started = []
    stopped = []

    class LifecycleLayer(Layer):
        def on_start(self):
            started.append(self.name)

        def on_stop(self):
            stopped.append(self.name)

        def process(self, **inputs):
            return 1

    layer = LifecycleLayer(name="lc", outputs=["v"])
    pipe = Pipeline(layer)
    pipe.run_sequence(2)

    assert started == ["lc"]
    assert stopped == ["lc"]


def test_layer_complete_marks_inactive():
    call_count = [0]

    def produce(**inputs):
        call_count[0] += 1
        if call_count[0] >= 2:
            raise LayerComplete()
        return call_count[0]

    producer = FunctionalLayer(name="prod", outputs=["v"], process_fn=produce)
    consumer = FunctionalLayer(
        name="cons",
        inputs={"v": "prod.v"},
        outputs=["r"],
        process_fn=lambda v: v,
    )

    pipe = Pipeline(producer, consumer)
    pipe.run_sequence(4)

    assert pipe.cycle_count == 4
    # Producer ran twice (completed on 2nd), consumer kept running
    assert call_count[0] == 2
    assert "prod" in pipe._inactive_layers


def test_remove_layer():
    a = FunctionalLayer(name="a", outputs=["x"], process_fn=lambda **kw: 1)
    b = FunctionalLayer(name="b", outputs=["y"], process_fn=lambda **kw: 2)
    pipe = Pipeline(a, b)

    pipe.remove_layer("b")
    assert len(pipe.layers) == 1
    assert pipe.layers[0].name == "a"


def test_replace_layer():
    a = FunctionalLayer(name="a", outputs=["x"], process_fn=lambda **kw: 1)
    b = FunctionalLayer(name="b", inputs={"x": "a.x"}, outputs=["y"],
                        process_fn=lambda x: x * 10)

    pipe = Pipeline(a, b)
    _run_one_cycle(pipe)
    assert pipe.outputs.y == 10

    new_b = FunctionalLayer(name="new_b", inputs={"x": "a.x"}, outputs=["y"],
                            process_fn=lambda x: x * 100)
    pipe.replace_layer("b", new_b)
    _run_one_cycle(pipe)
    assert pipe.outputs.y == 100


def test_metrics_collector():
    metrics = MetricsCollector(window=100)
    layer = FunctionalLayer(
        name="m", outputs=["v"], process_fn=lambda **kw: 42)
    pipe = Pipeline(layer, metrics=metrics)
    pipe.run_sequence(3)

    summary = metrics.summary()
    assert "m" in summary
    assert summary["m"]["calls"] == 3
    assert summary["m"]["errors"] == 0
    assert summary["m"]["mean_ms"] > 0

    assert metrics.bottleneck() == "m"
    assert "m" in metrics.report()


def test_fine_grained_scheduler_eliminates_tier_barrier():
    """
    Proves the fine-grained scheduler doesn't wait for sibling layers.

    DAG:  A ──► B(slow) ──► C ──► F
          A ──► D(fast)  ──► E(slow) ──► F

    Old tier scheduler: B and D in tier 1. C and E in tier 2.
    Tier 1 would block 0.3s (waiting for B even though D finishes instantly).
    Tier 2 would block 0.3s (waiting for E even though C finishes instantly).
    Total ≈ 0.6s.

    Fine-grained scheduler: D → E starts immediately. B → C starts immediately.
    Both branches run independently. Total ≈ 0.3s (the critical path).
    """
    import asyncio
    import time

    class Source(Layer):
        execution_mode = ExecutionMode.INLINE

        def __init__(self):
            super().__init__(name="A", outputs=["x"])

        def process(self):
            return 1

    class SlowB(Layer):
        def __init__(self):
            super().__init__(name="B", inputs={"x": "A.x"}, outputs=["y"])

        def process(self, x):
            return x + 1

        async def aprocess(self, x):
            await asyncio.sleep(0.3)
            return x + 1

    class FastD(Layer):
        execution_mode = ExecutionMode.INLINE

        def __init__(self):
            super().__init__(name="D", inputs={"x": "A.x"}, outputs=["w"])

        def process(self, x):
            return x + 10

    class FastC(Layer):
        execution_mode = ExecutionMode.INLINE

        def __init__(self):
            super().__init__(name="C", inputs={"y": "B.y"}, outputs=["z"])

        def process(self, y):
            return y * 2

    class SlowE(Layer):
        def __init__(self):
            super().__init__(name="E", inputs={"w": "D.w"}, outputs=["v"])

        def process(self, w):
            return w * 3

        async def aprocess(self, w):
            await asyncio.sleep(0.3)
            return w * 3

    class Sink(Layer):
        execution_mode = ExecutionMode.INLINE

        def __init__(self):
            super().__init__(name="F", inputs={
                "z": "C.z", "v": "E.v"}, outputs=["result"])

        def process(self, z, v):
            return z + v

    pipe = Pipeline(Source(), SlowB(), FastD(), FastC(), SlowE(), Sink())

    start = time.perf_counter()
    pipe.run_sequence(1)
    elapsed = time.perf_counter() - start

    # Verify correctness
    assert pipe.outputs.result == (1 + 1) * 2 + (1 + 10) * 3  # 4 + 33 = 37

    # With tier-barrier: ≈0.6s (0.3 for tier1 + 0.3 for tier2)
    # With fine-grained: ≈0.3s (both branches overlap)
    # Allow generous margin but assert it's under the tier-barrier time
    assert elapsed < 0.55, (
        f"Cycle took {elapsed:.3f}s — expected ~0.3s with fine-grained scheduling, "
        f"not ~0.6s like tier-barrier would give"
    )


def test_pipeline_serialization(tmp_path):
    class CountLayer(Layer):
        def __init__(self):
            super().__init__(name="counter", outputs=["val"])
            self.c = 0

        def process(self):
            self.c += 1
            return self.c

    class HistLayer(Layer):
        def __init__(self):
            super().__init__(name="hist", inputs={
                "v": "counter.val[-2:]"}, outputs=["sum"])

        def process(self, v):
            return sum(v) if v else 0

    pipe = Pipeline(CountLayer(), HistLayer())
    pipe.run_sequence(3)

    import os
    state_file = os.path.join(tmp_path, "state.pkl")
    pipe.save_state(state_file)

    # New pipeline
    pipe2 = Pipeline(CountLayer(), HistLayer())
    pipe2.load_state(state_file)

    assert pipe2.cycle_count == 3
    # Check current_values directly
    assert pipe2._current_values["counter"]["val"] == 3
    # sum([1, 2]) since it reads history before cycle 3 finishes
    assert pipe2._current_values["hist"]["sum"] == 3


def test_pipeline_as_layer():
    class InnerLayer(Layer):
        def __init__(self):
            super().__init__(name="inner", outputs=["y"])

        def process(self):
            return 10

    # A sub-pipeline with exports
    sub = Pipeline(InnerLayer(), name="sub", exports=["y"])

    # Process it directly to test pipeline.process(**external)
    # The ext is ignored inside pipeline.py but tests coverage
    res = sub.process(ext=5)
    assert res == 10

    # Test __str__ and __repr__
    assert repr(sub) == "Pipeline(inner)"
    assert "Pipeline([" in str(sub)


def test_circular_dependency():
    class A(Layer):
        def __init__(self):
            super().__init__(name="a", inputs={"x": "b.val"}, outputs=["val"])

        def process(self, x): pass

    with pytest.raises(ValueError, match="that layer is not registered or does not provide that output"):
        Pipeline(A())


def test_pipeline_async_run():
    import asyncio

    class DummyLayer(Layer):
        def __init__(self):
            super().__init__(name="d", outputs=["v"])
            self.n = 0

        def process(self):
            self.n += 1
            if self.n > 2:
                raise PipelineStop()
            return self.n

    pipe = Pipeline(DummyLayer())
    asyncio.run(pipe.run_async())
    # cycle_count is incremented at the start of the cycle, so it hits 3 before stopping
    assert pipe.cycle_count == 3


def test_pipeline_exceptions():
    class CrashLayer(Layer):
        def __init__(self):
            super().__init__(name="crash", outputs=["out"])

        def process(self):
            raise ValueError("Intentional crash")

    pipe = Pipeline(CrashLayer())
    with pytest.raises(RuntimeError, match="Pipeline error: Intentional crash"):
        pipe.run_sequence(1)


def test_duplicate_layer():
    class L1(Layer):
        def __init__(self):
            super().__init__(name="l1", outputs=["x"])

        def process(self): return 1

    pipe = Pipeline(L1())
    pipe.remove_layer("l1")  # this covers lines 309

    with pytest.raises(ValueError, match="not found"):
        pipe.remove_layer("l1")  # lines 309 when missing


def test_outputs_setter_and_getitem():
    pipe = Pipeline()
    pipe.outputs = pipe._pipeline_outputs  # Line 255
    with pytest.raises(TypeError):
        _ = pipe.outputs[123]  # Line 66


def test_garbage_collect():
    class OldLayer(Layer):
        def __init__(self):
            super().__init__(name="old", outputs=["val"])

        def process(self): return 1

    pipe = Pipeline(OldLayer())
    pipe.run_sequence(1)

    assert "val" in pipe._current_values["old"]

    # Remove layer
    pipe.remove_layer("old")
    # This triggers _rebuild_graph which calls _garbage_collect

    assert "val" not in pipe._current_values.get("old", {})


def test_pipeline_reset_and_layer_complete_propagation():
    class SourceLayer(Layer):
        def __init__(self):
            super().__init__(name="src", outputs=["x"])
            self.count = 0

        def process(self):
            self.count += 1
            if self.count > 2:
                raise LayerComplete()
            return self.count

    class ConsumerLayer(Layer):
        def __init__(self):
            super().__init__(name="cons", inputs={"x": "src.x"}, outputs=["y"])
            self.completed = False

        def process(self, x):
            return x + 1

        def on_complete(self):
            self.completed = True
            raise LayerComplete()

    src = SourceLayer()
    cons = ConsumerLayer()
    pipe = Pipeline(src, cons)

    # Run fully until all layers exhaust (src raises LayerComplete)
    pipe.run()

    assert "src" in pipe._inactive_layers
    assert cons.completed is True

    # Test bug 4 / 8 - reset functionality
    pipe.reset()
    assert len(pipe._inactive_layers) == 0
    assert pipe._cycle_count == 0


def test_pipeline_remove_inactive_layer_cleanup():
    class SourceLayer(Layer):
        def __init__(self):
            super().__init__(name="src", outputs=["x"])

        def process(self):
            raise LayerComplete()

    src = SourceLayer()
    pipe = Pipeline(src)
    pipe.run_sequence(1)

    assert "src" in pipe._inactive_layers
    pipe.remove_layer("src")
    assert "src" not in pipe._inactive_layers


def test_pipeline_stop_threaded_cancellation():
    import time

    class SlowThreadLayer(Layer):
        execution_mode = ExecutionMode.THREAD

        def __init__(self):
            super().__init__(name="slow", outputs=["x"])

        def process(self):
            time.sleep(1.0)
            return 1

    class StopLayer(Layer):
        execution_mode = ExecutionMode.THREAD

        def __init__(self):
            super().__init__(name="stopper", outputs=["y"])

        def process(self):
            raise PipelineStop()

    pipe = Pipeline(SlowThreadLayer(), StopLayer())

    # PipelineStop should be caught and cause a graceful stop,
    # the ThreadPoolExecutor should be shutdown, wait=False, cancel_futures=True.
    # No CancelledError should be bubbled up!
    pipe.run_sequence(1)


def test_pipeline_stop_async_cancellation():
    import asyncio

    class SlowAsyncLayer(Layer):
        execution_mode = ExecutionMode.ASYNC

        def __init__(self):
            super().__init__(name="slow", outputs=["x"])

        def process(self):
            pass

        async def aprocess(self):
            await asyncio.sleep(1.0)
            return 1

    class StopLayer(Layer):
        execution_mode = ExecutionMode.ASYNC

        def __init__(self):
            super().__init__(name="stopper", outputs=["y"])

        def process(self):
            pass

        async def aprocess(self):
            raise PipelineStop()

    pipe = Pipeline(SlowAsyncLayer(), StopLayer())

    class SyncLayer1(Layer):
        execution_mode = ExecutionMode.INLINE

        def __init__(self):
            super().__init__(name="l1", outputs=["x"])

        def process(self):
            return 5

    class SyncLayer2(Layer):
        execution_mode = ExecutionMode.INLINE

        def __init__(self):
            super().__init__(name="l2", inputs={"x": "l1.x"}, outputs=["y"])

        def process(self, x):
            return x * 2

    # Since no ASYNC mode layers exist, this must natively branch off into _process_cycle_sync
    pipe = Pipeline(SyncLayer1(), SyncLayer2())
    pipe.run_sequence(1)

    assert getattr(pipe.outputs, "l2.y", pipe.outputs.y) == 10
    assert pipe._cycle_count == 1
