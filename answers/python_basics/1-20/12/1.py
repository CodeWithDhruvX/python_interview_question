def handle_case_one():
    return "Handling case 1"

def handle_case_two():
    return "Handling case 2"

def handle_default():
    return "Handling default case"

def switch_case(case_value):
    switch = {
        'one': handle_case_one,
        'two': handle_case_two
    }
    return switch.get(case_value, handle_default)()

# Example usage
if __name__ == "__main__":
    print(switch_case('one'))     # Output: Handling case 1
    print(switch_case('three'))   # Output: Handling default case
