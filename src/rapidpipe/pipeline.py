from __future__ import annotations

import asyncio
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx

from rapidpipe.layer import (
    Layer, DependencyInfo, AccessType, ExecutionMode,
    LayerComplete, PipelineStop,
)
from rapidpipe.metrics import MetricsCollector
from rapidpipe.utils.visualizer import visualize_pipeline


# --------------------------------------------------------------------------- #
#  Unified history entry                                                       #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class HistoryEntry:
    value: Any
    cycle: int
    timestamp: float


# --------------------------------------------------------------------------- #
#  Pipeline-level outputs helper                                               #
# --------------------------------------------------------------------------- #
class PipelineOutputs:
    """
    Thin wrapper that tracks, for every *pipeline* output name, which layer
    most recently produced it.  Exposes outputs as attributes.
    """

    def __init__(self, pipeline: Pipeline):
        self._pipeline = pipeline
        # output_name -> layer_name
        self._output_providers: Dict[str, str] = {}
        self._available_outputs: set[str] = set()

    # ---- public interface -------------------------------------------------- #
    def __getattr__(self, output_name: str) -> Any:
        """Dynamically retrieve the latest value of a pipeline output."""
        if output_name.startswith("_"):
            raise AttributeError(output_name)
        if output_name not in self._output_providers:
            raise AttributeError(
                f"Pipeline output '{output_name}' is not available")
        return self._pipeline.get_value(
            self._output_providers[output_name], output_name
        )

    def __getitem__(self, key: str) -> Any:
        """Allow subscript access to pipeline outputs."""
        if not isinstance(key, str):
            raise TypeError("index must be a string")
        return getattr(self, key)

    def available_outputs(self) -> List[str]:
        """List all output names currently registered in the pipeline."""
        return list(self._available_outputs)

    # ---- internal helpers -------------------------------------------------- #
    def _register(self, output_name: str, layer_name: str) -> None:
        """Called when a layer is added to register its outputs."""
        self._available_outputs.add(output_name)
        if output_name not in self._output_providers:
            self._output_providers[output_name] = layer_name

    def _set_provider(self, output_name: str, layer_name: str) -> None:
        """Update the provider of an output when a layer executes."""
        self._output_providers[output_name] = layer_name

    def _remove_provider(self, output_name: str, layer_name: str) -> None:
        """Deregister an output provider when a layer is removed."""
        if self._output_providers.get(output_name) == layer_name:
            del self._output_providers[output_name]
        self._available_outputs.discard(output_name)

    def provider_of(self, output_name: str) -> Optional[str]:
        """Return the name of the layer providing the given output."""
        return self._output_providers.get(output_name)


# --------------------------------------------------------------------------- #
#  Dependency graph – single source of truth                                   #
# --------------------------------------------------------------------------- #
class DependencyGraph:
    """
    Directed graph whose nodes are (layer_name, output_name) tuples.
    Computes:
      - exact cycles / seconds each output must retain
      - cyclic-dependency validation (for current dependencies)
    """

    def __init__(self, pipeline: Pipeline):
        self._pipeline = pipeline
        self._graph: nx.DiGraph = nx.DiGraph()
        self._current_dependency_graph: nx.DiGraph = nx.DiGraph()
        self._storage: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: {"cycles": 0, "time_seconds": 0.0}
        )

    # ---- public API -------------------------------------------------------- #
    @property
    def graph(self) -> nx.DiGraph:
        """Get the full NetworkX Directed Acyclic Graph."""
        return self._graph

    def storage_requirement(self, layer: str, output: str) -> Dict[str, float]:
        """Get the calculated storage retention requirements for a given output."""
        return self._storage[(layer, output)]

    def rebuild(self) -> None:
        """
        Re-build the graph from scratch using current Pipeline state.
        This validates DAG topology, detects circular dependencies, and computes history windows.
        """
        self._graph.clear()
        self._current_dependency_graph.clear()
        self._storage.clear()

        layers = {lyr.name: lyr for lyr in self._pipeline.layers}

        # 1. Add nodes for every output of every layer
        for layer in layers.values():
            for out in layer.outputs:
                self._graph.add_node((layer.name, out))
                self._current_dependency_graph.add_node((layer.name, out))

        # 2. Add edges + compute storage needs
        for consumer in layers.values():
            for dep in consumer.parsed_dependencies.values():
                src_layer, src_out = self._resolve(dep)
                if src_layer is None:
                    continue

                edge = ((src_layer, src_out), (consumer.name, dep.output_name))
                self._graph.add_edge(*edge)

                if dep.access_type in (AccessType.CURRENT, AccessType.PIPELINE_CURRENT):
                    self._current_dependency_graph.add_edge(*edge)

                self._propagate_storage(
                    (src_layer, src_out),
                    cycles=self._cycles_needed(dep),
                    time_seconds=self._time_needed(dep),
                )

        # 3. Validate no cycles in current-value dependencies
        if not nx.is_directed_acyclic_graph(self._current_dependency_graph):
            cycles = list(nx.simple_cycles(self._current_dependency_graph))
            raise ValueError(
                f"Circular dependency detected in current values: {cycles}"
            )

    # ---- internal --------------------------------------------------------- #
    def _resolve(self, dep: DependencyInfo) -> Optional[Tuple[str, str]]:
        """Resolve a dependency specifier into a concrete layer and output name."""
        if dep.is_pipeline_dependency:
            layer = self._pipeline.outputs.provider_of(dep.output_name)
            return (layer, dep.output_name) if layer else None
        return (dep.layer_name, dep.output_name)

    @staticmethod
    def _cycles_needed(dep: DependencyInfo) -> int:
        """Calculate the max cycle index required by a dependency."""
        if dep.access_type in (AccessType.INDEXED, AccessType.PIPELINE_INDEXED):
            return dep.index or 0
        if dep.access_type in (AccessType.INDEX_RANGE, AccessType.PIPELINE_INDEX_RANGE):
            return dep.index_start or 0
        return 0

    @staticmethod
    def _time_needed(dep: DependencyInfo) -> float:
        """Calculate the time window required by a dependency in seconds."""
        if dep.access_type in (AccessType.TIME_RANGE, AccessType.PIPELINE_TIME_RANGE):
            return dep.time_start or 0.0
        return 0.0

    def _propagate_storage(
        self,
        node: Tuple[str, str],
        cycles: int,
        time_seconds: float,
    ) -> None:
        """Recursively propagate the maximum storage requirements up the graph."""
        current = self._storage[node]
        new_cycles = max(current["cycles"], cycles)
        new_time = max(current["time_seconds"], time_seconds)
        if new_cycles == current["cycles"] and new_time == current["time_seconds"]:
            return

        self._storage[node] = {"cycles": new_cycles, "time_seconds": new_time}

        for pred in self._graph.predecessors(node):
            self._propagate_storage(pred, cycles, time_seconds)


# --------------------------------------------------------------------------- #
#  Main Pipeline class                                                         #
# --------------------------------------------------------------------------- #
class Pipeline(Layer):
    """
    The orchestrator that executes layers topologically, manages their histories,
    and supports acting as a standalone nested Layer inside parent pipelines.
    """

    def __init__(
        self,
        *layers: Layer,
        name: Optional[str] = None,
        inputs: Optional[Dict[str, str]] = None,
        exports: Optional[List[str]] = None,
        max_history_size: int = 1000,
        metrics: Optional[MetricsCollector] = None,
    ):
        """
        Initialize the Pipeline.

        Args:
            *layers: Layer instances to register in the pipeline.
            name: Name of the pipeline (used when nested).
            inputs: Exposes dependencies from parent pipelines when nested.
            exports: Defines which outputs should be exported to the parent pipeline.
            max_history_size: Global upper limit on deque sizes for tracking history.
            metrics: Optional MetricsCollector instance for latency/error tracking.
        """
        # Layer init (Pipeline-as-Layer)
        super().__init__(name=name, inputs=inputs, outputs=exports or [])

        self._layers: List[Layer] = []
        self._max_history_size = max_history_size
        self._metrics = metrics

        # Unified history store
        self._current_values: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(deque)
        )

        self._cycle_count = 0
        self._running = False

        # Fine-grained dependency scheduling (replaces tier-barrier approach)
        # layer -> set of producer layer names
        self._layer_producers: Dict[str, set] = {}
        # layer -> set of consumer layer names
        self._layer_consumers: Dict[str, set] = {}

        # Layers marked as complete (LayerComplete)
        self._inactive_layers: set[str] = set()

        # External inputs for sub-pipeline mode
        self._external_inputs: Dict[str, Any] = {}

        # Graph-based bookkeeping
        self._graph = DependencyGraph(self)
        # Override the outputs from Layer base class with PipelineOutputs
        self._output_defs = self._output_names  # preserve for sub-pipeline exports
        self._pipeline_outputs = PipelineOutputs(self)

        for layer in layers:
            self.add_layer(layer)

    # --------------------------------------------------------------------- #
    #  Pipeline outputs (override the Layer .outputs for attribute access)    #
    # --------------------------------------------------------------------- #
    @property
    def outputs(self) -> PipelineOutputs:
        """Get the dynamic pipeline outputs interface."""
        return self._pipeline_outputs

    @outputs.setter
    def outputs(self, value):
        # Allow setting during __init__ from Layer base
        self._pipeline_outputs = value

    # ---- public layer management ------------------------------------------ #
    @property
    def layers(self) -> List[Layer]:
        """Return the current list of registered layers."""
        return self._layers

    @property
    def cycle_count(self) -> int:
        """Return the number of cycles executed so far."""
        return self._cycle_count

    def add_layer(self, layer: Layer) -> None:
        """
        Register a new layer into the pipeline.

        Validates that its dependencies are satisfied by previously added layers
        if it uses 'current cycle' dependencies, then triggers graph rebuild.

        Args:
            layer: The initialized Layer instance.
        """
        if not isinstance(layer, Layer):
            raise TypeError("Only Layer instances can be added")

        if any(l.name == layer.name for l in self._layers):
            raise ValueError(
                f"A layer with name '{layer.name}' already exists.")

        # Validate current-value dependencies
        for dep in layer.parsed_dependencies.values():
            if dep.access_type == AccessType.PIPELINE_CURRENT:
                is_provided = any(
                    dep.output_name in lyr.outputs for lyr in self._layers
                )
                if not is_provided:
                    raise ValueError(
                        f"Layer '{layer.name}' has a dependency on pipeline output '{dep.output_name}', "
                        "but no layer already in the pipeline provides it."
                    )
            elif dep.access_type == AccessType.CURRENT:
                is_provided = any(
                    lyr.name == dep.layer_name and dep.output_name in lyr.outputs
                    for lyr in self._layers
                )
                if not is_provided:
                    raise ValueError(
                        f"Layer '{layer.name}' has a dependency on '{dep.layer_name}.{dep.output_name}', "
                        "but that layer is not registered or does not provide that output."
                    )

        layer.add_to_pipeline(self)
        self._layers.append(layer)

        for out in layer.outputs:
            self._pipeline_outputs._register(out, layer.name)

        if self._metrics:
            self._metrics.register(layer.name)

        self._graph.rebuild()
        self._recompute_scheduling()
        self._garbage_collect()

    def remove_layer(self, name: str) -> None:
        """
        Remove a layer from the pipeline by its identifier name.

        This will issue a UserWarning that any layers remaining in the pipeline
        that depended on this layer will now continuously receive `None`.

        Args:
            name: String name of the layer to remove.
        """
        idx = next((i for i, l in enumerate(
            self._layers) if l.name == name), None)
        if idx is None:
            raise ValueError(f"Layer '{name}' not found")
        old = self._layers.pop(idx)
        for out in old.outputs:
            self._pipeline_outputs._remove_provider(out, old.name)
        self._graph.rebuild()
        self._recompute_scheduling()
        self._garbage_collect()

        import warnings
        warnings.warn(
            f"Layer '{name}' removed. Layers depending on its outputs may receive None.", UserWarning)

    def replace_layer(self, name: str, new_layer: Layer) -> None:
        """
        Hot-swap a layer at runtime with a new implementation.

        Automatically preserves the recorded historical values for any output
        keys that both the old layer and the new layer share. Useful for
        dynamic re-routing or upgrading logic without restarting the pipeline.

        Args:
            name: The name of the existing layer to replace.
            new_layer: The new Layer instance to insert in its place.
        """
        idx = next((i for i, l in enumerate(
            self._layers) if l.name == name), None)
        if idx is None:
            raise ValueError(f"Layer '{name}' not found")
        old_layer = self._layers[idx]
        shared_outputs = set(old_layer.outputs) & set(new_layer.outputs)

        new_layer.add_to_pipeline(self)
        self._layers[idx] = new_layer
        # Transfer history for shared outputs
        for out in shared_outputs:
            if old_layer.name in self._history and out in self._history[old_layer.name]:
                self._history[new_layer.name][out] = self._history[old_layer.name][out]

        for out in old_layer.outputs:
            self._pipeline_outputs._remove_provider(out, old_layer.name)
        for out in new_layer.outputs:
            self._pipeline_outputs._register(out, new_layer.name)

        if self._metrics:
            self._metrics.register(new_layer.name)

        self._graph.rebuild()
        self._recompute_scheduling()

    # ---- DAG scheduling --------------------------------------------------- #
    def _recompute_scheduling(self) -> None:
        """
        Build per-layer producer/consumer maps from the current-cycle dependency
        subgraph. Used by the fine-grained task-graph scheduler.
        """
        producers: Dict[str, set] = {layer.name: set()
                                     for layer in self._layers}
        consumers: Dict[str, set] = {layer.name: set()
                                     for layer in self._layers}

        # Also build a full map including historical dependencies for effective mode calculation
        full_consumers: Dict[str, set] = {
            layer.name: set() for layer in self._layers}

        for consumer in self._layers:
            for dep in consumer.parsed_dependencies.values():
                src = self._graph._resolve(dep)
                if src and src[0]:
                    full_consumers[src[0]].add(consumer.name)
                    if dep.access_type in (AccessType.CURRENT, AccessType.PIPELINE_CURRENT):
                        producers[consumer.name].add(src[0])
                        consumers[src[0]].add(consumer.name)

        self._layer_producers = producers
        self._layer_consumers = consumers

        # Compute effective execution modes for AUTO layers
        all_layers = {layer.name for layer in self._layers}
        descendants: Dict[str, set] = {name: set() for name in all_layers}

        for start_layer in all_layers:
            visited = set()
            queue = deque(full_consumers[start_layer])
            while queue:
                curr = queue.popleft()
                if curr not in visited:
                    visited.add(curr)
                    queue.extend(full_consumers[curr])
            descendants[start_layer] = visited

        ancestors: Dict[str, set] = {name: set() for name in all_layers}
        for start_layer, descs in descendants.items():
            for desc in descs:
                ancestors[desc].add(start_layer)

        self._effective_execution_modes = {}
        for layer in self._layers:
            mode = layer.execution_mode
            has_custom_aprocess = hasattr(type(layer), 'aprocess') and getattr(
                type(layer), 'aprocess') is not getattr(Layer, 'aprocess', None)

            if mode == ExecutionMode.INLINE:
                effective = ExecutionMode.INLINE
            elif mode == ExecutionMode.ASYNC or (mode == ExecutionMode.AUTO and has_custom_aprocess):
                effective = ExecutionMode.ASYNC
            elif mode == ExecutionMode.THREAD:
                effective = ExecutionMode.THREAD
            else:
                # AUTO mode with sync process
                concurrent_peers = all_layers - \
                    descendants[layer.name] - \
                    ancestors[layer.name] - {layer.name}
                if not concurrent_peers:
                    effective = ExecutionMode.INLINE
                else:
                    effective = ExecutionMode.THREAD

            self._effective_execution_modes[layer.name] = effective

    # ---- run methods ------------------------------------------------------- #

    def run(self) -> None:
        """
        Blocking, synchronous entry point to start the continuous execution loop.
        Creates a new event loop and runs until stopped or a PipelineStop is raised.
        """
        asyncio.run(self._async_run_loop())

    async def run_async(self) -> None:
        """
        Asynchronous execution loop for use when already inside an event loop 
        (e.g. Jupyter, FastAPI, Quart). Runs continuously.
        """
        await self._async_run_loop()

    def run_sequence(self, n: int) -> None:
        """
        Run exactly N execution cycles synchronously, then stop.

        Args:
            n: Number of pipeline cycles to compute.
        """
        asyncio.run(self._run_n(n))

    def run_until(self, predicate: Callable[[Pipeline], bool]) -> None:
        """
        Run synchronously until a given callable predicate returns True.

        Args:
            predicate: A function that takes the pipeline instance and returns a boolean.
        """
        asyncio.run(self._run_until(predicate))

    async def _async_run_loop(self) -> None:
        self._running = True
        self._call_on_start()
        try:
            while self._running:
                try:
                    await self._process_cycle()
                except PipelineStop:
                    break
                except Exception as e:
                    raise RuntimeError(f"Pipeline error: {e}") from e
        finally:
            self._running = False
            self._call_on_stop()

    async def _run_n(self, n: int) -> None:
        self._running = True
        self._call_on_start()
        try:
            for _ in range(n):
                if not self._running:
                    break
                try:
                    await self._process_cycle()
                except PipelineStop:
                    break
                except Exception as e:
                    raise RuntimeError(f"Pipeline error: {e}") from e
        finally:
            self._running = False
            self._call_on_stop()

    async def _run_until(self, predicate: Callable[[Pipeline], bool]) -> None:
        self._running = True
        self._call_on_start()
        try:
            while self._running:
                try:
                    await self._process_cycle()
                except PipelineStop:
                    break
                except Exception as e:
                    raise RuntimeError(f"Pipeline error: {e}") from e
                if predicate(self):
                    break
        finally:
            self._running = False
            self._call_on_stop()

    def stop(self) -> None:
        """Gracefully interrupt and stop the running pipeline."""
        self._running = False

    # ---- lifecycle --------------------------------------------------------- #
    def _call_on_start(self) -> None:
        for layer in self._layers:
            layer.on_start()

    def _call_on_stop(self) -> None:
        for layer in self._layers:
            layer.on_stop()

    # ---- core processing --------------------------------------------------- #
    async def _process_cycle(self) -> None:
        """
        Execute one full pipeline cycle using fine-grained task-graph scheduling.

        Instead of grouping layers into tiers and waiting for each tier to
        complete, this scheduler launches every layer as soon as all of its
        current-cycle producers have finished.  This eliminates the tier-barrier
        penalty in heterogeneous DAGs.

        Algorithm:
            1. Initialize a pending-count for each layer (= number of unfinished
               producers it depends on).
            2. Seed the ready-set with all layers whose pending-count is 0.
            3. Launch ready layers concurrently as asyncio Tasks.
            4. When a task finishes, commit its outputs to the snapshot, then
               decrement the pending-count of every consumer.  Any consumer
               whose count reaches 0 is scheduled immediately.
            5. Repeat until all layers are done or PipelineStop is raised.
        """
        self._cycle_count += 1
        cycle_ts = time.time()

        # Snapshot — shared mutable dict updated as layers complete
        snapshot: Dict[str, Dict[str, Any]] = {
            layer_name: dict(outputs)
            for layer_name, outputs in self._current_values.items()
        }
        all_produced: Dict[str, Dict[str, Any]] = {}
        name_to_layer = {l.name: l for l in self._layers}

        # Pending-count: how many producers each layer is still waiting for
        pending: Dict[str, int] = {
            name: len(producers)
            for name, producers in self._layer_producers.items()
        }

        pipeline_stop = False
        in_flight: set = set()          # set of asyncio.Task
        task_to_layer: Dict[asyncio.Task, Layer] = {}

        def _schedule_layer(layer: Layer) -> None:
            """Create an asyncio.Task for a single layer."""
            task = asyncio.ensure_future(self._invoke_layer(layer, snapshot))
            in_flight.add(task)
            task_to_layer[task] = layer

        # Seed: schedule every layer with zero pending producers
        for layer in self._layers:
            if layer.name in self._inactive_layers:
                # Treat inactive as instantly done — unblock consumers
                for consumer_name in self._layer_consumers.get(layer.name, set()):
                    pending[consumer_name] -= 1
                continue
            if not self._should_run(layer, snapshot):
                for consumer_name in self._layer_consumers.get(layer.name, set()):
                    pending[consumer_name] -= 1
                continue
            if pending.get(layer.name, 0) == 0:
                _schedule_layer(layer)

        # Drain in-flight tasks one-by-one as they complete
        while in_flight:
            done, in_flight = await asyncio.wait(
                in_flight, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                layer = task_to_layer.pop(task)

                # Handle exceptions
                exc = task.exception()
                if exc is not None:
                    if isinstance(exc, PipelineStop):
                        pipeline_stop = True
                        # Cancel remaining tasks
                        for t in in_flight:
                            t.cancel()
                        in_flight.clear()
                        break
                    raise exc

                result = task.result()
                if result is not None:
                    outputs = self._unpack_result(layer, result)
                    snapshot[layer.name] = {
                        **snapshot.get(layer.name, {}), **outputs
                    }
                    all_produced[layer.name] = outputs

                # Unblock consumers of this layer
                for consumer_name in self._layer_consumers.get(layer.name, set()):
                    pending[consumer_name] -= 1
                    if pending[consumer_name] == 0:
                        consumer = name_to_layer[consumer_name]
                        if (
                            consumer.name not in self._inactive_layers
                            and self._should_run(consumer, snapshot)
                        ):
                            _schedule_layer(consumer)

            if pipeline_stop:
                break

        self._apply_produced(all_produced)
        self._commit_history(all_produced, cycle_ts)

        if pipeline_stop:
            raise PipelineStop()

    def _should_run(self, layer: Layer, snapshot: Dict) -> bool:
        """Check should_run with resolved inputs from snapshot."""
        inputs = {
            k: self._resolve_from_snapshot(dep, snapshot)
            for k, dep in layer.parsed_dependencies.items()
        }
        should = layer.should_run(**inputs)
        if not should and self._metrics:
            self._metrics.record_skip(layer.name)
        return should

    async def _invoke_layer(self, layer: Layer, snapshot: Dict) -> Any:
        inputs = {
            k: self._resolve_from_snapshot(dep, snapshot)
            for k, dep in layer.parsed_dependencies.items()
        }

        try:
            if self._metrics:
                with self._metrics.measure(layer.name):
                    result = await self._dispatch(layer, inputs)
                if result is not None:
                    self._metrics.record_outputs(
                        layer.name, self._unpack_result(layer, result)
                    )
                return result
            return await self._dispatch(layer, inputs)
        except LayerComplete:
            self._inactive_layers.add(layer.name)
            return None
        except PipelineStop:
            self._running = False
            raise

    async def _dispatch(self, layer: Layer, inputs: Dict) -> Any:
        mode = self._effective_execution_modes.get(
            layer.name, ExecutionMode.INLINE)

        if mode == ExecutionMode.INLINE:
            return layer.process(**inputs)

        if mode == ExecutionMode.ASYNC:
            return await layer.aprocess(**inputs)

        # THREAD mode
        return await asyncio.to_thread(layer.process, **inputs)

    def _resolve_from_snapshot(self, dep: DependencyInfo, snapshot: Dict) -> Any:
        """Resolve a dependency from the tier snapshot (for current) or history."""
        layer, out = self._graph._resolve(dep)
        if layer is None:
            return None

        if dep.access_type == AccessType.CURRENT:
            return snapshot.get(layer, {}).get(out)

        if dep.access_type == AccessType.PIPELINE_CURRENT:
            provider_layer = self._pipeline_outputs.provider_of(out)
            if provider_layer:
                return snapshot.get(provider_layer, {}).get(out)
            return None

        if dep.access_type in (AccessType.INDEXED, AccessType.PIPELINE_INDEXED):
            hist = self._history[layer][out]
            idx = dep.index
            if idx is not None and idx > 0 and len(hist) >= idx:
                return hist[-idx].value
            return None

        if dep.access_type in (AccessType.INDEX_RANGE, AccessType.PIPELINE_INDEX_RANGE):
            hist = list(self._history[layer][out])
            start_idx = -dep.index_start if dep.index_start is not None else None
            end_idx = -(dep.index_end +
                        1) if dep.index_end is not None else None
            if end_idx == 0:
                end_idx = None
            return [e.value for e in hist[start_idx:end_idx]]

        if dep.access_type in (AccessType.TIME_RANGE, AccessType.PIPELINE_TIME_RANGE):
            now = time.time()
            hist = self._history[layer][out]
            start_time_threshold = now - (dep.time_start or 0)
            end_time_threshold = now - (dep.time_end or 0)
            return [
                e.value for e in hist
                if end_time_threshold <= e.timestamp <= start_time_threshold
            ]

        raise ValueError(f"Unsupported access type: {dep.access_type}")

    def _unpack_result(self, layer: Layer, result: Any) -> Dict[str, Any]:
        if len(layer.outputs) == 1:
            return {layer.outputs[0]: result}
        if isinstance(result, (tuple, list)):
            return dict(zip(layer.outputs, result))
        return {layer.outputs[0]: result}

    def _apply_produced(self, produced: Dict[str, Dict[str, Any]]) -> None:
        for layer_name, outs in produced.items():
            for out, value in outs.items():
                self._current_values[layer_name][out] = value
                self._pipeline_outputs._set_provider(out, layer_name)

    def _commit_history(self, produced: Dict[str, Dict[str, Any]], cycle_ts: float) -> None:
        for layer_name, outs in produced.items():
            for out, value in outs.items():
                req = self._graph.storage_requirement(layer_name, out)
                needs_cycles = req["cycles"] > 0
                needs_time = req["time_seconds"] > 0
                if not needs_cycles and not needs_time:
                    continue

                entry = HistoryEntry(
                    value=value, cycle=self._cycle_count, timestamp=cycle_ts)
                dq = self._history[layer_name][out]
                dq.append(entry)

                # Prune: keep entries that satisfy cycle OR time retention
                max_entries = self._max_history_size
                while len(dq) > max_entries:
                    dq.popleft()
                if needs_cycles:
                    while len(dq) > req["cycles"] + 1:
                        # Only prune if the oldest entry also doesn't satisfy time
                        if needs_time and dq[0].timestamp >= cycle_ts - req["time_seconds"]:
                            break
                        dq.popleft()
                if needs_time:
                    while dq and dq[0].timestamp < cycle_ts - req["time_seconds"]:
                        if needs_cycles and len(dq) <= req["cycles"] + 1:
                            break
                        dq.popleft()

    # ---- introspection helpers -------------------------------------------- #
    def get_value(self, layer: str | Layer, output: str) -> Any:
        return self._current_values[
            layer.name if isinstance(layer, Layer) else layer
        ].get(output)

    def get_storage_requirement(self, layer: str, output: str) -> Dict[str, float]:
        return self._graph.storage_requirement(layer, output)

    def show_graph(self, name: str = "", free: bool = True, metrics_mode: bool = False) -> None:
        visualize_pipeline(self, name, free=free)

    # ---- state serialization ---------------------------------------------- #
    def save_state(self, path: str) -> None:
        """Pickle current values and history. Layer code is not saved."""
        state = {
            "cycle_count": self._cycle_count,
            "current_values": dict(self._current_values),
            "history": {
                layer: {out: list(entries) for out, entries in outs.items()}
                for layer, outs in self._history.items()
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, path: str) -> None:
        """Restore values and history. Pipeline structure must match."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._cycle_count = state["cycle_count"]
        self._current_values.update(state["current_values"])
        for layer, outs in state["history"].items():
            for out, entries in outs.items():
                self._history[layer][out] = deque(
                    entries, maxlen=self._max_history_size)

    # ---- garbage collection ----------------------------------------------- #
    def _garbage_collect(self) -> None:
        active = set(self._graph.graph.nodes)
        for layer, out_dict in list(self._current_values.items()):
            for out in list(out_dict):
                if (layer, out) not in active:
                    del out_dict[out]
        for layer, out_dict in list(self._history.items()):
            for out in list(out_dict):
                if (layer, out) not in active:
                    del out_dict[out]

    # ---- Pipeline-as-Layer process() -------------------------------------- #
    async def aprocess(self, **external_inputs) -> Any:
        """
        Coroutine-safe execution of a single cycle, seeding pipeline-level deps from external_inputs.
        Returns exported outputs in declaration order.
        """
        self._external_inputs = external_inputs
        await self._process_cycle()
        return self._export_results()

    def process(self, **external_inputs) -> Any:
        """
        Run one internal cycle sequentially. If an event loop is already running, 
        use aprocess() instead to avoid loop conflicts.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            raise RuntimeError(
                "Cannot call synchronous process() from within a running event loop. "
                "Use await pipeline.aprocess() instead."
            )

        self._external_inputs = external_inputs
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._process_cycle())
        finally:
            loop.close()

        return self._export_results()

    def _export_results(self) -> Any:
        export_names = self._output_names if isinstance(
            self._output_names, list) else []
        if not export_names:
            return None

        results = [
            self._current_values.get(
                self._pipeline_outputs.provider_of(n), {}
            ).get(n)
            for n in export_names
        ]
        return results if len(results) > 1 else results[0]

    # ---- string helpers ---------------------------------------------------- #
    def __repr__(self) -> str:
        return f"Pipeline({', '.join(lyr.name for lyr in self.layers)})"

    def __str__(self) -> str:
        lines = ["Pipeline(["]
        for lyr in self.layers:
            lines.append(f"  {lyr.name}")
        lines.append("])")
        return "\n".join(lines)
