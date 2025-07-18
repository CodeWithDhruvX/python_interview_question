import gc

class Node:
    def __init__(self, name):
        self.name = name
        self.ref = None

    def __del__(self):
        print(f"Node {self.name} destroyed")

def circular_reference_demo():
    gc.set_debug(gc.DEBUG_UNCOLLECTABLE)

    a = Node("A")
    b = Node("B")
    a.ref = b
    b.ref = a  # Circular reference created

    del a
    del b  # Objects still not destroyed due to cycle

    print("\nRunning garbage collector...\n")
    unreachable = gc.collect()  # Force collection of cycles

    print(f"Unreachable objects: {unreachable}")
    print("Garbage:", gc.garbage)

circular_reference_demo()
