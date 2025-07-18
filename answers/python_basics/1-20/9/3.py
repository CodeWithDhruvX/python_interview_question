def print_even_numbers(numbers):
    print("Even numbers in the list:")
    for num in numbers:
        if num % 2 != 0:
            continue  # Skip odd numbers
        print(num)

if __name__ == "__main__":
    sample_list = [10, 15, 22, 33, 40, 55]
    print_even_numbers(sample_list)
