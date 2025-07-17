def modify_sequence(seq):
    try:
        seq[0] = 99
        print("Modified:", seq)
    except TypeError as e:
        print("Error:", e)

# List example
my_list = [1, 2, 3]
print("Original List:", my_list)
modify_sequence(my_list)

# Tuple example
my_tuple = (1, 2, 3)
print("\nOriginal Tuple:", my_tuple)
modify_sequence(my_tuple)
