"""
Example 8 — Sub-Pipelines (Pipeline-as-Layer)
==============================================
Demonstrates composing complex pipelines out of smaller sub-pipelines.
Since Pipeline inherits from Layer, it can be added to another Pipeline.
"""
from rapidpipe import Pipeline, Layer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ── Layers for Data Ingestion Sub-Pipeline ────────────────────────────────── #

class FileScanner(Layer):
    def __init__(self):
        super().__init__(name="scanner", outputs=["filepath"])
        self._n = 0

    def process(self) -> str:
        self._n += 1
        return f"/data/file_{self._n}.txt"


class FileLoader(Layer):
    def __init__(self):
        super().__init__(name="loader", inputs={
            "path": "scanner.filepath"}, outputs=["raw_data"])

    def process(self, path: str) -> str:
        return f"content_of_({path})"


# ── Layers for Processing Sub-Pipeline ────────────────────────────────────── #

class TextCleaner(Layer):
    def __init__(self):
        super().__init__(name="cleaner", inputs={
            "text": "raw_data"}, outputs=["clean_text"])

    def process(self, text: str) -> str:
        return text.upper()


class Tokenizer(Layer):
    def __init__(self):
        super().__init__(name="tokenizer", inputs={
            "text": "cleaner.clean_text"}, outputs=["tokens"])

    def process(self, text: str) -> list:
        return text.split("_")


# ── Main Pipeline Layer ───────────────────────────────────────────────────── #

class OutputPrinter(Layer):
    def __init__(self):
        super().__init__(name="printer", inputs={
            "tokens": "tokenizer.tokens"}, outputs=["status"])

    def process(self, tokens: list) -> str:
        print(f"  Received tokens: {tokens}")
        return "printed"


# ── Build Composition ─────────────────────────────────────────────────────── #

print("=" * 60)
print("Example 8 — Modular Pipelines")
print("=" * 60)
print()

# 1. Ingestion Pipeline
ingestion_pipe = Pipeline(
    FileScanner(),
    FileLoader(),
    name="ingestion_pipeline"
)

print("Running Ingestion Pipeline...")
raw_data_history = []
for _ in range(3):
    ingestion_pipe.run_sequence(1)
    raw_data_history.append(ingestion_pipe.outputs.raw_data)

print(f"Ingested {len(raw_data_history)} files.")

# 2. Processing Pipeline
# We create a simple source layer that yields the ingested data


class DataFeeder(Layer):
    def __init__(self, data: list):
        super().__init__(name="feeder", outputs=["raw_data"])
        self.data = iter(data)

    def process(self):
        try:
            return next(self.data)
        except StopIteration:
            from rapidpipe.layer import LayerComplete
            raise LayerComplete()


processing_pipe = Pipeline(
    DataFeeder(raw_data_history),
    TextCleaner(),
    Tokenizer(),
    OutputPrinter(),
    name="processing_pipeline"
)

print("\nRunning Processing Pipeline...")
processing_pipe.run_sequence(5)  # Runs until data is exhausted

print("\nProcessing Pipeline Cycles:", processing_pipe.cycle_count)
print("Processing Pipeline Layers:", [l.name for l in processing_pipe.layers])
print("=" * 60)

processing_pipe.show_graph("08 — Modular Pipelines")
