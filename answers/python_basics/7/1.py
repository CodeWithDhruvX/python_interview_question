# Global variable
counter = 10

def increase():
    # Local variable
    counter = 5
    counter += 1
    print("Inside function (local):", counter)

increase()
print("Outside function (global):", counter)
