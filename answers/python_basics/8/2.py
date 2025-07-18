def ticket_price(age):
    if age < 5:
        return "Free"
    elif age < 18:
        return "Child Ticket - ₹100"
    elif age < 60:
        return "Adult Ticket - ₹200"
    else:
        return "Senior Citizen Ticket - ₹150"

# Sample ages to test
ages = [3, 10, 25, 65]

for age in ages:
    print(f"Age: {age}, Price: {ticket_price(age)}")
