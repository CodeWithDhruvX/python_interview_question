from collections import defaultdict

def group_by_type(data):
    grouped = defaultdict(list)
    for item in data:
        grouped[type(item).__name__].append(item)
    
    print("Grouped Data by Type:\n")
    for dtype, items in grouped.items():
        print(f"{dtype}: {items}")

if __name__ == "__main__":
    data = [1, 2.0, "hello", True, None, {"x": 1}, [1, 2], "world", (5, 6), {10, 20}]
    group_by_type(data)
