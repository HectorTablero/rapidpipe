"""
Example 1 — Basic Pipeline
===========================
A linear pipeline with three layers: a data source, a transformer,
and a formatter. Demonstrates:

  • Defining layers with process()
  • Current-value dependencies (layer.output and pipeline-level)
  • run_sequence(n) to execute a fixed number of cycles
  • Accessing pipeline outputs
"""
from rapidpipe import Pipeline, Layer, PipelineStop
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ── Layer definitions ──────────────────────────────────────────────────────── #

class SensorLayer(Layer):
    """Simulates a sensor that produces incrementing readings."""

    def __init__(self):
        super().__init__(name="sensor", outputs=["temperature", "humidity"])
        self._step = 0

    def process(self) -> tuple:
        self._step += 1
        temp = 20.0 + self._step * 0.5
        hum = 45.0 + self._step * 1.2
        return temp, hum


class ConverterLayer(Layer):
    """Converts Celsius to Fahrenheit and computes a comfort index."""

    def __init__(self):
        super().__init__(
            name="converter",
            inputs={
                "celsius": "sensor.temperature",
                "humidity": "sensor.humidity",
            },
            outputs=["fahrenheit", "comfort_index"],
        )

    def process(self, celsius, humidity) -> tuple:
        fahr = celsius * 9 / 5 + 32
        # Simple comfort index: lower is better
        comfort = abs(celsius - 22) + abs(humidity - 50) * 0.3
        return round(fahr, 1), round(comfort, 2)


class ReporterLayer(Layer):
    """Formats a human-readable report string."""

    def __init__(self):
        super().__init__(
            name="reporter",
            inputs={
                "temp": "temperature",              # pipeline-level dependency
                "fahr": "converter.fahrenheit",      # direct layer dependency
                "comfort": "converter.comfort_index",
            },
            outputs=["report"],
        )

    def process(self, temp, fahr, comfort) -> str:
        return f"{temp:.1f}°C ({fahr}°F) | comfort={comfort}"


# ── Build & run ────────────────────────────────────────────────────────────── #

pipe = Pipeline(
    SensorLayer(),
    ConverterLayer(),
    ReporterLayer(),
)

N_CYCLES = 6
pipe.run_sequence(N_CYCLES)

# ── Print results ──────────────────────────────────────────────────────────── #

print("=" * 60)
print("Example 1 — Basic Pipeline")
print("=" * 60)
print(f"Cycles run: {pipe.cycle_count}")
print(f"Available outputs: {pipe.outputs.available_outputs()}")
print()
print(f"  temperature  = {pipe.outputs.temperature}")
print(f"  humidity     = {pipe.outputs.humidity}")
print(f"  fahrenheit   = {pipe.outputs.fahrenheit}")
print(f"  comfort_idx  = {pipe.outputs.comfort_index}")
print(f"  report       = {pipe.outputs.report}")
print()
print(f"Pipeline structure: {pipe}")
print("=" * 60)

# ── Visualize ──────────────────────────────────────────────────────────────── #
pipe.show_graph("01 — Basic Pipeline")
