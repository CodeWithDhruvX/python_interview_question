counter = 0  # Global variable

def increment():
    global counter  # Declare we're using the global variable
    counter += 1
    print("Inside function:", counter)

for _ in range(5):
    increment()

print("Outside function:", counter)
