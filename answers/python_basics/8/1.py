def get_grade(score):
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    else:
        return "F"

# Example usage
scores = [95, 83, 72, 67, 40]
for s in scores:
    print(f"Score: {s}, Grade: {get_grade(s)}")
