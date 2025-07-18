from collections import defaultdict

def group_names_by_first_letter(names):
    grouped = defaultdict(list)

    for name in names:
        first_letter = name[0].upper()
        grouped[first_letter].append(name)

    return dict(grouped)

# Example usage
names = ["Alice", "Ankit", "Bob", "Bhuvan", "Charlie", "Chirag"]
result = group_names_by_first_letter(names)
print(result)
