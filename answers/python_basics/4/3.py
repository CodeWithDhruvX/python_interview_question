# Snippet 3: Using a custom class
class Student:
    def __init__(self, name):
        self.name = name

s1 = Student("Rahul")
s2 = Student("Rahul")

print("s1 == s2:", s1 == s2)
print("s1 is s2:", s1 is s2)

# Assigning same reference
s3 = s1
print("s1 == s3:", s1 == s3)
print("s1 is s3:", s1 is s3)
