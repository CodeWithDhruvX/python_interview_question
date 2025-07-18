import types
from collections.abc import Iterable, Mapping

def classify_data_types(value):
    if isinstance(value, str):
        return "String"
    elif isinstance(value, bool):
        return "Boolean"
    elif isinstance(value, int):
        return "Integer"
    elif isinstance(value, float):
        return "Float"
    elif isinstance(value, bytes):
        return "Bytes"
    elif isinstance(value, Mapping):
        return "Dictionary"
    elif isinstance(value, Iterable):
        return "Iterable (List/Tuple/Set/etc)"
    elif value is None:
        return "NoneType"
    elif isinstance(value, types.FunctionType):
        return "Function"
    else:
        return "Unknown Type"

# Example usage
sample_values = [123, 3.14, "Hello", [1, 2], {"a": 1}, True, None, (1, 2), b"data", lambda x: x * 2]

for val in sample_values:
    print(f"Value: {val!r:<15} => Type: {classify_data_types(val)}")
