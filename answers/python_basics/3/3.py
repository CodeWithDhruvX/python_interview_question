def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b if b != 0 else "Cannot divide by zero"

operations = {
    "add": add,
    "sub": subtract,
    "mul": multiply,
    "div": divide
}

# Example usage
op = "mul"
a, b = 6, 3
result = operations[op](a, b)
print(f"Result of {op} operation: {result}")
