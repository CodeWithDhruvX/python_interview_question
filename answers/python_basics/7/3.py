def make_counter():
    count = 0  # Local variable in enclosing scope

    def increment():
        nonlocal count  # Refers to 'count' in the outer function
        count += 1
        return count

    return increment

counter = make_counter()

for _ in range(5):
    print("Counter:", counter())
