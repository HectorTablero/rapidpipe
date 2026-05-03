from __future__ import annotations
import os
import time
import asyncio
import numpy as np
import pytest
import concurrent.futures
from rapidpipe import Pipeline, Layer, ExecutionMode, MetricsCollector, LayerComplete, PipelineStop
from rapidpipe.numba_layer import NumbaLayer
from numba import prange

# --------------------------------------------------------------------------- #
#  Example 1 Layers                                                           #
# --------------------------------------------------------------------------- #

class Ex1Sensor(Layer):
    def __init__(self):
        super().__init__(name="sensor", outputs=["temperature", "humidity"])
        self._step = 0
    def process(self):
        self._step += 1
        return 20.0 + self._step * 0.5, 45.0 + self._step * 1.2

class Ex1Converter(Layer):
    def __init__(self):
        super().__init__(name="converter", inputs={"celsius": "sensor.temperature", "humidity": "sensor.humidity"}, outputs=["fahrenheit", "comfort_index"])
    def process(self, celsius, humidity):
        fahr = celsius * 9 / 5 + 32
        comfort = abs(celsius - 22) + abs(humidity - 50) * 0.3
        return round(fahr, 1), round(comfort, 2)

class Ex1Reporter(Layer):
    def __init__(self):
        super().__init__(name="reporter", inputs={"temp": "temperature", "fahr": "converter.fahrenheit", "comfort": "converter.comfort_index"}, outputs=["report"])
    def process(self, temp, fahr, comfort):
        return f"{temp:.1f}°C ({fahr}°F) | comfort={comfort}"

def test_example_01_basic():
    pipe = Pipeline(Ex1Sensor(), Ex1Converter(), Ex1Reporter())
    pipe.run_sequence(6)
    assert pipe.cycle_count == 6
    assert pipe.outputs.temperature == 23.0
    assert pipe.outputs.humidity == 52.2
    assert pipe.outputs.fahrenheit == 73.4
    assert pipe.outputs.comfort_index == 1.66
    assert pipe.outputs.report == "23.0°C (73.4°F) | comfort=1.66"

# --------------------------------------------------------------------------- #
#  Example 2 Layers                                                           #
# --------------------------------------------------------------------------- #

class Ex2Camera(Layer):
    execution_mode = ExecutionMode.INLINE
    def __init__(self):
        super().__init__(name="camera", outputs=["frame"])
        self._i = 0
    def process(self):
        self._i += 1
        return f"frame_{self._i}"

class Ex2Depth(Layer):
    execution_mode = ExecutionMode.INLINE
    def __init__(self):
        super().__init__(name="depth", outputs=["depth_map"])
        self._i = 0
    def process(self):
        self._i += 1
        return f"depth_{self._i}"

class Ex2IMU(Layer):
    execution_mode = ExecutionMode.INLINE
    def __init__(self):
        super().__init__(name="imu", outputs=["orientation"])
        self._i = 0
    def process(self):
        self._i += 1
        return f"orient_{self._i}"

class Ex2Fusion(Layer):
    execution_mode = ExecutionMode.INLINE
    def __init__(self):
        super().__init__(name="fusion", inputs={"frame": "camera.frame", "depth": "depth.depth_map", "orient": "imu.orientation"}, outputs=["fused"])
    def process(self, frame, depth, orient):
        return f"FUSED({frame}, {depth}, {orient})"

class Ex2Display(Layer):
    execution_mode = ExecutionMode.INLINE
    def __init__(self):
        super().__init__(name="display", inputs={"data": "fusion.fused"}, outputs=["rendered"])
    def process(self, data):
        return f"[DISPLAY] {data}"

def test_example_02_dag_parallelism():
    pipe = Pipeline(Ex2Camera(), Ex2Depth(), Ex2IMU(), Ex2Fusion(), Ex2Display())
    pipe.run_sequence(3)
    assert pipe.cycle_count == 3
    assert pipe.outputs.fused == "FUSED(frame_3, depth_3, orient_3)"
    assert pipe.outputs.rendered == "[DISPLAY] FUSED(frame_3, depth_3, orient_3)"

# --------------------------------------------------------------------------- #
#  Example 3 Layers                                                           #
# --------------------------------------------------------------------------- #

class Ex3Signal(Layer):
    def __init__(self):
        super().__init__(name="signal", outputs=["value"])
        self._n = 0
    def process(self):
        self._n += 10
        return self._n

class Ex3Previous(Layer):
    def __init__(self):
        super().__init__(name="prev", inputs={"lagged": "signal.value[-2]"}, outputs=["lagged"])
    def process(self, lagged): return lagged

class Ex3MovingAvg(Layer):
    def __init__(self):
        super().__init__(name="moving_avg", inputs={"window": "signal.value[-3:]"}, outputs=["average"])
    def process(self, window):
        if not window: return None
        return round(sum(window) / len(window), 2)

def test_example_03_history():
    pipe = Pipeline(Ex3Signal(), Ex3Previous(), Ex3MovingAvg())
    
    # Check storage requirements
    req = pipe.get_storage_requirement('signal', 'value')
    assert req['cycles'] == 3
    
    # Run 6 cycles and check values
    for i in range(1, 7):
        pipe.run_sequence(1)
        if i == 1:
            assert pipe.outputs.value == 10
            assert pipe.outputs.lagged is None
            assert pipe.outputs.average is None
        elif i == 4:
            assert pipe.outputs.value == 40
            assert pipe.outputs.lagged == 20
            # window = [10, 20, 30] -> avg = 20.0
            assert pipe.outputs.average == 20.0
    
    assert pipe.outputs.value == 60
    assert pipe.outputs.lagged == 40
    assert pipe.outputs.average == 40.0

# --------------------------------------------------------------------------- #
#  Example 4 Layers                                                           #
# --------------------------------------------------------------------------- #

class Ex4Source(Layer):
    def __init__(self):
        super().__init__(name="source", outputs=["value"])
        self._data = iter(range(1, 6))
        self.started = False
        self.stopped = False
    def on_start(self): self.started = True
    def on_stop(self): self.stopped = True
    def process(self):
        try: return next(self._data)
        except StopIteration: raise LayerComplete()

class Ex4RateLimited(Layer):
    def __init__(self):
        super().__init__(name="rate_limited", inputs={"value": "source.value"}, outputs=["processed"])
        self.run_count = 0
    def should_run(self, **inputs):
        return self._pipeline.cycle_count % 2 == 1
    def process(self, value):
        self.run_count += 1
        return (value or 0) * 100

class Ex4Consumer(Layer):
    def __init__(self):
        super().__init__(name="consumer", inputs={"data": "rate_limited.processed"}, outputs=["final"])
    def process(self, data): return f"consumed({data})"

def test_example_04_conditional_lifecycle():
    pipe = Pipeline(Ex4Source(), Ex4RateLimited(), Ex4Consumer())
    pipe.run_sequence(8)
    assert pipe.cycle_count == 8
    assert 'source' in pipe._inactive_layers
    assert pipe.layers[1].run_count == 4
    assert pipe.outputs.final == "consumed(500)"
    assert pipe.layers[0].started is True
    assert pipe.layers[0].stopped is True

# --------------------------------------------------------------------------- #
#  Example 5 Layers                                                           #
# --------------------------------------------------------------------------- #

class Ex5Fast(Layer):
    execution_mode = ExecutionMode.INLINE
    def __init__(self):
        super().__init__(name="fast_producer", outputs=["data"])
        self._n = 0
    def process(self):
        self._n += 1
        return self._n

class Ex5Medium(Layer):
    execution_mode = ExecutionMode.INLINE
    def __init__(self):
        super().__init__(name="medium_transform", inputs={"data": "fast_producer.data"}, outputs=["transformed"])
    def process(self, data):
        _ = sum(range(1000)) # Small delay
        return data * 2.5

class Ex5Slow(Layer):
    execution_mode = ExecutionMode.INLINE
    def __init__(self):
        super().__init__(name="slow_aggregator", inputs={"val": "medium_transform.transformed"}, outputs=["result"])
    def process(self, val):
        _ = sum(range(5000)) # Larger delay
        return f"result={val}"

def test_example_05_metrics():
    metrics = MetricsCollector(window=200)
    pipe = Pipeline(Ex5Fast(), Ex5Medium(), Ex5Slow(), metrics=metrics)
    pipe.run_sequence(100)
    
    assert pipe.outputs.result == "result=250.0"
    summary = metrics.summary()
    assert "slow_aggregator" in summary
    assert summary["slow_aggregator"]["calls"] == 100
    assert metrics.bottleneck() == "slow_aggregator"

# --------------------------------------------------------------------------- #
#  Example 6 Layers                                                           #
# --------------------------------------------------------------------------- #

class Ex6Counter(Layer):
    def __init__(self):
        super().__init__(name="counter", outputs=["n"])
        self._n = 0
    def process(self):
        self._n += 1
        return self._n

class Ex6Multiplier(Layer):
    def __init__(self, factor=2, name="multiplier"):
        super().__init__(name=name, inputs={"n": "counter.n"}, outputs=["product"])
        self.factor = factor
    def process(self, n): return n * self.factor

class Ex6Logger(Layer):
    def __init__(self):
        super().__init__(name="logger", inputs={"product": "multiplier.product"}, outputs=["log"])
    def process(self, product):
        return f"cycle {self._pipeline.cycle_count}: product={product}"

def test_example_06_hot_swap():
    pipe = Pipeline(Ex6Counter(), Ex6Multiplier(factor=2), Ex6Logger())
    
    # Phase 1
    pipe.run_until(lambda p: p.outputs.product >= 10)
    assert pipe.cycle_count == 5
    assert pipe.outputs.product == 10
    
    # Phase 2
    pipe.replace_layer("multiplier", Ex6Multiplier(factor=10, name="multiplier"))
    pipe.run_sequence(3)
    assert pipe.cycle_count == 8
    assert pipe.outputs.product == 80
    
    # Phase 3
    with pytest.warns(UserWarning, match="Layer 'logger' removed"):
        pipe.remove_layer("logger")
    
    pipe.run_sequence(2)
    assert pipe.cycle_count == 10
    assert pipe.outputs.product == 100
    assert len(pipe.layers) == 2

# --------------------------------------------------------------------------- #
#  Example 7 Layers                                                           #
# --------------------------------------------------------------------------- #

def heavy_computation(data: np.ndarray, iterations: int) -> np.ndarray:
    result = np.zeros_like(data)
    for i in range(data.shape[0]):
        val = data[i]
        for _ in range(iterations):
            val = (val * 1.05) ** 0.99
        result[i] = val
    return result

class Ex7Source(Layer):
    def __init__(self): super().__init__(name="source", outputs=["data"])
    def process(self): return np.array([0.5, 0.1, 0.9])

class Ex7Python(Layer):
    def __init__(self, iterations: int):
        super().__init__(name="python_compute", inputs={"data": "source.data"}, outputs=["result"])
        self.iterations = iterations
    def process(self, data): return heavy_computation(data, self.iterations)

class Ex7Numba(NumbaLayer, nopython=True, nogil=True):
    def __init__(self, iterations: int):
        super().__init__(name="numba_compute", inputs={"data": "source.data"}, outputs=["result"])
        self.iterations = iterations
    @staticmethod
    def kernel(data, iterations):
        result = np.zeros_like(data)
        for i in range(data.shape[0]):
            val = data[i]
            for _ in range(iterations):
                val = (val * 1.05) ** 0.99
            result[i] = val
        return result
    def process(self, data): return self._compiled_kernel(data, self.iterations)

class Ex7ParallelNumba(NumbaLayer, parallel=True, nopython=True, nogil=True):
    def __init__(self, iterations: int):
        super().__init__(name="parallel_numba_compute", inputs={"data": "source.data"}, outputs=["result"])
        self.iterations = iterations
    @staticmethod
    def kernel(data, iterations):
        result = np.zeros_like(data)
        for i in prange(data.shape[0]):
            val = data[i]
            for _ in range(iterations):
                val = (val * 1.05) ** 0.99
            result[i] = val
        return result
    def process(self, data): return self._compiled_kernel(data, self.iterations)

def test_example_07_numba_perf():
    pipe = Pipeline(Ex7Source(), Ex7Python(iterations=10), Ex7Numba(iterations=10), Ex7ParallelNumba(iterations=10))
    pipe.run_sequence(5)
    py_res = pipe.get_value("python_compute", "result")
    nb_res = pipe.get_value("numba_compute", "result")
    pnb_res = pipe.get_value("parallel_numba_compute", "result")
    assert np.allclose(py_res, nb_res)
    assert np.allclose(nb_res, pnb_res)

# --------------------------------------------------------------------------- #
#  Example 8 Layers                                                           #
# --------------------------------------------------------------------------- #

class Ex8Source(Layer):
    def __init__(self):
        super().__init__(name="source", outputs=["tick"])
        self.count = 0
    def process(self):
        self.count += 1
        return self.count
    def get_state(self): return {"count": self.count}
    def set_state(self, state): self.count = state.get("count", 0)

class Ex8Multiplier(Layer):
    def __init__(self):
        super().__init__(name="multiplier", inputs={"val": "source.tick", "scale": "factor"}, outputs=["result"])
    def process(self, val, scale): return val * scale

class Ex8FactorProvider(Layer):
    def __init__(self): super().__init__(name="factor_provider", outputs=["external_factor"])
    def process(self): return 10

class Ex8Sink(Layer):
    def __init__(self): super().__init__(name="sink", inputs={"x": "sub_pipe.result"}, outputs=[])
    def process(self, x): pass

def test_example_08_pipeline_as_layer():
    sub = Pipeline(Ex8Source(), Ex8Multiplier(), name="sub_pipe", inputs={"factor": "external_factor"}, exports=["result"])
    parent = Pipeline(Ex8FactorProvider(), sub, Ex8Sink(), name="parent_pipe")
    
    parent.run_sequence(3)
    assert parent.outputs.result == 30
    
    state_file = "test_nested_state.pkl"
    parent.save_state(state_file)
    parent.run_sequence(2)
    assert parent.outputs.result == 50
    
    parent.load_state(state_file)
    parent.run_sequence(1)
    assert parent.outputs.result == 40
    
    if os.path.exists(state_file): os.remove(state_file)

# --------------------------------------------------------------------------- #
#  Example 9 Layers                                                           #
# --------------------------------------------------------------------------- #

class Ex9Trigger(Layer):
    def __init__(self):
        super().__init__(name="trigger", outputs=["cycle_id"])
        self.n = 0
    def process(self):
        self.n += 1
        return self.n

class Ex9API_A(Layer):
    def __init__(self): super().__init__(name="api_a", inputs={"id": "trigger.cycle_id"}, outputs=["response"])
    def process(self, id): pass
    async def aprocess(self, id):
        await asyncio.sleep(0.1)
        return f"A-{id}"

class Ex9API_B(Layer):
    def __init__(self): super().__init__(name="api_b", inputs={"id": "trigger.cycle_id"}, outputs=["response"])
    def process(self, id): pass
    async def aprocess(self, id):
        await asyncio.sleep(0.1)
        return f"B-{id}"

class Ex9Combiner(Layer):
    def __init__(self): super().__init__(name="combiner", inputs={"resp_a": "api_a.response", "resp_b": "api_b.response"}, outputs=["result"])
    def process(self, resp_a, resp_b): return f"{resp_a} | {resp_b}"

def test_example_09_async():
    pipe = Pipeline(Ex9Trigger(), Ex9API_A(), Ex9API_B(), Ex9Combiner())
    start = time.time()
    pipe.run_sequence(3)
    duration = time.time() - start
    assert pipe.outputs.result == "A-3 | B-3"
    # Each cycle should take ~0.1s due to parallelism, total ~0.3s. Sequential would be ~0.6s.
    assert duration < 0.5 

# --------------------------------------------------------------------------- #
#  Example 10 Layers                                                          #
# --------------------------------------------------------------------------- #

class Ex10State(Layer):
    def __init__(self):
        super().__init__(name="counter", outputs=["n"])
        self.val = 0
    def process(self):
        self.val += 1
        return self.val
    def get_state(self): return {"val": self.val}
    def set_state(self, state): self.val = state.get("val", 0)

class Ex10History(Layer):
    def __init__(self): super().__init__(name="history_tracker", inputs={"window": "counter.n[-3:]"}, outputs=["sum"])
    def process(self, window): return sum(window) if window else 0

def test_example_10_serialization():
    pipe_a = Pipeline(Ex10State(), Ex10History())
    pipe_a.run_sequence(5)
    assert pipe_a.outputs.n == 5
    assert pipe_a.outputs.sum == 9 # 2+3+4
    
    state_file = "test_state_10.pkl"
    pipe_a.save_state(state_file)
    
    pipe_b = Pipeline(Ex10State(), Ex10History())
    pipe_b.load_state(state_file)
    assert pipe_b.cycle_count == 5
    assert pipe_b.outputs.n == 5
    
    pipe_b.run_sequence(2)
    assert pipe_b.cycle_count == 7
    assert pipe_b.outputs.n == 7
    assert pipe_b.outputs.sum == 15 # 4+5+6
    
    if os.path.exists(state_file): os.remove(state_file)

# --------------------------------------------------------------------------- #
#  Example 11 Layers                                                          #
# --------------------------------------------------------------------------- #

class Ex11Source(Layer):
    def __init__(self):
        super().__init__(name="A", outputs=["x"])
        self._n = 0
    def process(self):
        self._n += 1
        return self._n

class Ex11SlowB(Layer):
    def __init__(self): super().__init__(name="B", inputs={"x": "A.x"}, outputs=["y"])
    def process(self, x):
        time.sleep(0.1)
        return x * 10

class Ex11FastC(Layer):
    execution_mode = ExecutionMode.INLINE
    def __init__(self): super().__init__(name="C", inputs={"y": "B.y"}, outputs=["z"])
    def process(self, y): return y + 1

class Ex11FastD(Layer):
    execution_mode = ExecutionMode.INLINE
    def __init__(self): super().__init__(name="D", inputs={"x": "A.x"}, outputs=["w"])
    def process(self, x): return x * 100

class Ex11SlowE(Layer):
    def __init__(self): super().__init__(name="E", inputs={"w": "D.w"}, outputs=["v"])
    def process(self, w):
        time.sleep(0.1)
        return w + 1

class Ex11Sink(Layer):
    def __init__(self): super().__init__(name="F", inputs={"z": "C.z", "v": "E.v"}, outputs=["result"])
    def process(self, z, v): return z + v

def test_example_11_fine_grained():
    metrics = MetricsCollector()
    pipe = Pipeline(Ex11Source(), Ex11SlowB(), Ex11FastC(), Ex11FastD(), Ex11SlowE(), Ex11Sink(), metrics=metrics)
    
    start = time.perf_counter()
    pipe.run_sequence(1)
    elapsed = time.perf_counter() - start
    
    assert pipe.outputs.result == 112
    # Should take ~0.1s, not ~0.2s
    assert elapsed < 0.18

# --------------------------------------------------------------------------- #
#  Example 12 Layers                                                          #
# --------------------------------------------------------------------------- #

class Ex12VectorSource(Layer):
    def __init__(self): super().__init__(name="source", outputs=["data"])
    def process(self): return np.array([1.0, 2.0, 3.0, 4.0])

class Ex12ThresholdConfig(Layer):
    def __init__(self, threshold: float):
        super().__init__(name="config", outputs=["threshold"])
        self.threshold = threshold
    def process(self): return self.threshold

class Ex12ClipLayer(NumbaLayer):
    def __init__(self):
        super().__init__(name="clipper", inputs={"my_array": "source.data", "ceiling": "config.threshold"}, outputs=["result"])
    @staticmethod
    def kernel(my_array, ceiling):
        res = np.empty_like(my_array)
        for i in range(my_array.shape[0]):
            res[i] = my_array[i] if my_array[i] < ceiling else ceiling
        return res

def test_example_12_numba_default():
    pipe = Pipeline(Ex12VectorSource(), Ex12ThresholdConfig(2.5), Ex12ClipLayer())
    pipe.run_sequence(1)
    expected = np.array([1.0, 2.0, 2.5, 2.5])
    assert np.array_equal(pipe.outputs.result, expected)

# --------------------------------------------------------------------------- #
#  Legacy Consistency Tests                                                   #
# --------------------------------------------------------------------------- #

def test_parallel_vs_inline_consistency():
    def create_complex_dag(mode: ExecutionMode):
        class ConstSource(Layer):
            def __init__(self, name, val):
                super().__init__(name=name, outputs=["out"])
                self.val = val
            def process(self): return self.val + self._pipeline.cycle_count
        
        class Adder(Layer):
            def __init__(self, name, input_specs: dict, mode=ExecutionMode.AUTO):
                self.execution_mode = mode
                super().__init__(name=name, inputs=input_specs, outputs=["sum"])
            def process(self, **kwargs):
                if any(v is None for v in kwargs.values()): return None
                return sum(kwargs.values())

        s1, s2, s3, s4 = ConstSource("s1", 1), ConstSource("s2", 2), ConstSource("s3", 3), ConstSource("s4", 4)
        a1 = Adder("a1", {"v1": "s1.out", "v2": "s2.out"}, mode=mode)
        a2 = Adder("a2", {"v1": "s3.out", "v2": "s4.out"}, mode=mode)
        m1 = Adder("m1", {"v1": "a1.sum", "v2": "a2.sum"}, mode=mode)
        return Pipeline(s1, s2, s3, s4, a1, a2, m1)

    p_par = create_complex_dag(ExecutionMode.THREAD)
    p_inl = create_complex_dag(ExecutionMode.INLINE)
    p_par.run_sequence(10)
    p_inl.run_sequence(10)
    assert p_par.outputs.sum == p_inl.outputs.sum == 50
