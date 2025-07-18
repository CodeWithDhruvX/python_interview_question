from collections import Counter

def summarize_data_types(data):
    type_names = [type(item).__name__ for item in data]
    counts = Counter(type_names)
    print("Data Type Summary:\n")
    for dtype, count in counts.items():
        print(f"{dtype}: {count}")

if __name__ == "__main__":
    mixed_data = [1, 2.5, "hi", None, True, False, {"a": 1}, [1, 2], (3, 4), {1, 2}, "world"]
    summarize_data_types(mixed_data)
