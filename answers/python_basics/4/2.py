# Snippet 2: Comparing lists
list1 = [1, 2, 3]
list2 = [1, 2, 3]

print("list1 == list2:", list1 == list2)  # Value equality
print("list1 is list2:", list1 is list2)  # Identity check

# Now aliasing list1 to list3
list3 = list1

print("list1 == list3:", list1 == list3)
print("list1 is list3:", list1 is list3)
