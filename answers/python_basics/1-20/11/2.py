def filter_even_numbers(numbers):
    result = []
    for num in numbers:
        if num % 2 != 0:
            continue
        result.append(num)
        print(f"Even number added: {num}")
    print("Filtered even numbers:", result)

filter_even_numbers([1, 2, 3, 4, 5, 6, 7])
