# RapidPipe: Advanced Data Orchestration Framework

RapidPipe is a high-performance, asynchronous-capable, and history-aware pipeline execution framework for Python. It provides an elegant, expressive syntax for defining directed acyclic graphs (DAGs) of computational layers, complete with built-in temporal windowing, dynamic task-graph scheduling, and live visualization.

Built from the ground up to support both synchronous (CPU-bound) and asynchronous (I/O-bound) operations natively, RapidPipe eliminates the boilerplate of thread pools and event loops, letting you focus entirely on your transformation logic.

---

## 🌟 Key Features

- **Intuitive Dependency Resolution**: Declare inputs via powerful string semantics (e.g., `"signal.value[-3:]"` or `"camera.frame"`) and let the framework automatically calculate topology and data-retention requirements.
- **Fine-Grained Task Scheduling**: Pipeline cycles run using an optimized `asyncio` task-graph. Layers launch the exact microsecond their dependencies resolve, eliminating synchronous layer-barrier penalties.
- **Smart Execution Modes (`AUTO`)**: The pipeline performs static analysis on your DAG to automatically assign the most efficient threading or inlining constraints. Isolated layers execute inline to avoid overhead, while functionally independent branches inherently parallelize.
- **Built-in Temporal History**: Read from past cycles trivially. The pipeline automatically maintains rolling deques for each output, sized strictly to what downstream layers need.
- **Interactive D3.js Visualization**: Instantly generate physics-based, interactive HTML maps of your architecture, tracking specific execution modes, topological layout, and (optionally) live latency metrics.
- **Nested Pipelines**: Use a full `Pipeline` as a standard `Layer` inside another pipeline, easily building highly composable and reusable sub-architectures.
- **Hot-swapping & Modifiability**: Dynamically add, remove, or hot-swap layers at runtime using `.replace_layer()` with zero downtime, preserving relevant historical states.

---

## 🏗️ Core Concepts

### 1. Layers (`Layer`)

A layer is the fundamental unit of computation. Every layer must define its expected `inputs` and `outputs` during `__init__`, and implement a `process()` (and/or `async def aprocess()`) method.

```python
from rapidpipe import Pipeline, Layer

class SensorLayer(Layer):
    def __init__(self):
        super().__init__(name="sensor", outputs=["temperature"])

    def process(self):
        return 22.5  # Returns to "temperature"

class LoggerLayer(Layer):
    def __init__(self):
        super().__init__(name="logger", inputs={"temp": "sensor.temperature"})

    def process(self, temp):
        print(f"Logged temperature: {temp}")
```

### 2. Execution Modes

Layers can specify an `execution_mode`:

- **`AUTO` (Default):** Dynamically optimizes. If the layer has an `aprocess()`, it executes as an `async` task. If symmetric concurrent branches exist, it executes in a `THREAD`. If it sequentially blocks all execution (no peers), it safely executes `INLINE`.
- **`ASYNC`:** Enforces execution on the `asyncio` event loop (requires `aprocess()`). Ideal for I/O waits, sockets, and API calls.
- **`THREAD`:** Enforces execution onto a concurrent ThreadPool. Ideal for CPU-bound computations that release the GIL (e.g., Pandas/Numpy operations).
- **`INLINE`:** Forces blocking, synchronous execution directly on the main orchestration thread. Best for trivial logic where threading overhead costs more than the process itself.

### 3. Smart History Variables

Access historical execution output immediately simply by slicing dependencies:

- **Current value:** `"layer.variable"` (Creates a strict cyclic dependency).
- **Previous value:** `"layer.variable[-1]"` (Refers to 1 cycle ago).
- **Index range (Windowing):** `"layer.variable[-3:]"` (List array of the last 3 values).
- **Time range:** `"layer.variable[-2.5s:]"` (Values from the last 2.5 seconds).

_Note: Sliced dependencies (`[-1]`, `[-3:]`) are considered chronological requirements, meaning they do not block execution synchronously in the current DAG topological cycle._

---

## 🚀 Advanced Capabilities

### Interactive Diagnostics

Diagnose bottlenecks and verify DAG structures visually:

```python
pipe.show_graph(name="My Pipeline", free=True)
```

Creates a dynamic local HTML/D3.js instance coloring `.layer` topologies by dependencies!

### State Serialization

Stop a pipeline, save its history bounds to disk, and revive it effortlessly:

```python
pipe.save_state("monday_run.pkl")
# ... later, or in a fresh node:
pipe.load_state("monday_run.pkl")
```

### Numba Integration

Included `numba_layer.py` seamlessly allows ultra-fast JIT-compiled matrix mathematics natively pipelined without escaping the Python DAG.

### Performance Metrics & Profiling

Pass a `MetricsCollector` to the Pipeline, run benchmarks, and examine `.summary()` to inspect median latencies, jitter, and P99 latency times inside of layers.

---

## 📂 Project Structure

```text
rapidpipe/
├── src/rapidpipe/         # Core Framework (Layer, Pipeline, Metrics)
│   └── utils/             # Visualization tools (D3.js generation)
├── examples/              # Educational blueprints & implementation demos
│   ├── 01_basic_pipeline.py
│   ├── 02_dag_parallelism.py
│   ├── 03_history_and_windowing.py
│   └── ...
└── tests/                 # Pytest validation suites ensuring deterministic
    └── ...                  pipeline features and error boundaries.
```

## 🛠 Usage & Quickstart

1. **Prerequisites:** Python 3.10+ highly recommended.
2. **Installation:** Since this relies closely on native `asyncio` and `nx` graphing patterns (with visualization HTMLs), ensure your virtual environment installs the base standard tooling via `pyproject.toml`.
3. **Examples:** Navigate to `examples/` and run any script to see it in action!
    ```powershell
    python examples/01_basic_pipeline.py
    # or
    uv run examples/01_basic_pipeline.py
    ```
4. **Testing:** Run the internal suite using standard `pytest`:
    ```powershell
    pytest tests/
    ```
