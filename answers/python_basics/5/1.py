import sys
import gc

class Sample:
    def __del__(self):
        print("Sample object destroyed")

def reference_count_demo():
    obj = Sample()
    print("Initial refcount:", sys.getrefcount(obj))

    alias1 = obj
    alias2 = obj
    print("After 2 more refs:", sys.getrefcount(obj))

    del alias1
    print("After deleting one alias:", sys.getrefcount(obj))

    del alias2
    print("After deleting second alias:", sys.getrefcount(obj))

    del obj
    print("Now object should be destroyed")

    gc.collect()  # Force garbage collection just to be sure

reference_count_demo()
