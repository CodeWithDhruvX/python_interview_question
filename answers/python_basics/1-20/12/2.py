def switch_case(case_value):
    match case_value:
        case 1:
            return "Case 1 matched"
        case 2:
            return "Case 2 matched"
        case 3 | 4:
            return "Case 3 or 4 matched"
        case _:
            return "Default case"

# Example usage
if __name__ == "__main__":
    print(switch_case(2))   # Output: Case 2 matched
    print(switch_case(5))   # Output: Default case
