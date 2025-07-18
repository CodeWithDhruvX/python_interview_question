def explore_data_types():
    data_examples = {
        "Integer": 42,
        "Float": 3.14,
        "String": "Hello, Python!",
        "Boolean": True,
        "List": [1, 2, 3],
        "Tuple": (1, 2, 3),
        "Set": {1, 2, 3},
        "Dict": {"a": 1, "b": 2},
        "NoneType": None,
        "Bytes": b"hello"
    }

    for dtype, example in data_examples.items():
        print(f"{dtype:<10} | Value: {example!r:<20} | type(): {type(example).__name__:<10} | isinstance: {isinstance(example, type(example))}")

if __name__ == "__main__":
    explore_data_types()
