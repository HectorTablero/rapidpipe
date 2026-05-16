"""
Example 13 — Context Managers
==============================
Demonstrates how to use Pipeline as both a synchronous and asynchronous
context manager to automate lifecycle hooks (on_start/on_stop) and 
resource cleanup.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rapidpipe import Pipeline, Layer


class LifecycleLayer(Layer):
    """A layer that tracks its own lifecycle status."""
    def __init__(self, name: str):
        super().__init__(name=name, outputs=["status"])
        self.status = "INITIALIZED"


    def on_start(self):
        print(f"  [Layer {self.name}] on_start() called")
        self.status = "STARTED"

    def on_stop(self):
        print(f"  [Layer {self.name}] on_stop() called")
        self.status = "STOPPED"

    def process(self):
        return self.status


def sync_demo():
    print("-" * 60)
    print("DEMO: Synchronous Context Manager")
    print("-" * 60)
    
    layer = LifecycleLayer("SyncNode")
    pipe = Pipeline(layer)
    
    print(f"Status before 'with': {layer.status}")
    
    with pipe:
        print(f"Status inside 'with': {layer.status}")
        # We can run cycles manually or use pipe.run()
        pipe.run_sequence(2)
        print("Completed 2 cycles.")
        
    print(f"Status after 'with': {layer.status}")
    print()


async def async_demo():
    print("-" * 60)
    print("DEMO: Asynchronous Context Manager")
    print("-" * 60)
    
    layer = LifecycleLayer("AsyncNode")
    pipe = Pipeline(layer)
    
    print(f"Status before 'async with': {layer.status}")
    
    async with pipe:
        print(f"Status inside 'async with': {layer.status}")
        # Run in background for a moment then stop
        print("Starting async loop in background...")
        task = asyncio.create_task(pipe.run_async())
        
        await asyncio.sleep(0.1)
        print("Signaling stop...")
        pipe.stop()
        await task

        
    print(f"Status after 'async with': {layer.status}")
    print()


print("=" * 60)
print("Example 13 — Context Managers")
print("=" * 60)
print()

# 1. Sync Demo
sync_demo()

# 2. Async Demo
try:
    asyncio.run(async_demo())
except KeyboardInterrupt:
    pass

print("=" * 60)
