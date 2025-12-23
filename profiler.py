import time
from collections import OrderedDict

class Profiler:
    def __init__(self):
        self.events = OrderedDict()
        self.start_times = {}

    def start(self, name):
        """Start a timer for a named event."""
        self.start_times[name] = time.time()
        print(f"[Profiler] Starting: {name}...")

    def stop(self, name):
        """Stop the timer for a named event and record duration."""
        if name in self.start_times:
            duration = time.time() - self.start_times.pop(name)
            self.events[name] = duration
            print(f"[Profiler] Finished: {name} ({duration:.2f}s)")
            return duration
        return 0

    def report(self):
        """Print a formatted performance report."""
        print("\n" + "="*50)
        print("PERFORMANCE REPORT")
        print("="*50)
        total_time = 0
        for name, duration in self.events.items():
            print(f"{name:<35}: {duration:.4f}s")
            total_time += duration
        print("-" * 50)
        print(f"{'Total Execution Time':<35}: {total_time:.4f}s")
        print("="*50 + "\n")
