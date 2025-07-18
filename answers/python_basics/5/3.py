import tracemalloc
import time

def memory_intensive_function():
    data = [str(i) * 1000 for i in range(100000)]
    time.sleep(1)
    return data

def profile_memory():
    tracemalloc.start()

    start_snapshot = tracemalloc.take_snapshot()
    memory_intensive_function()
    end_snapshot = tracemalloc.take_snapshot()

    stats = end_snapshot.compare_to(start_snapshot, 'lineno')
    print("\nTop memory usage lines:")
    for stat in stats[:5]:
        print(stat)

profile_memory()
