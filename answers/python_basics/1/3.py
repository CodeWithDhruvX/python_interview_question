# Using tuple as a dictionary key
location_data = {
    (28.6139, 77.2090): "New Delhi",
    (19.0760, 72.8777): "Mumbai"
}

print("City at (28.6139, 77.2090):", location_data[(28.6139, 77.2090)])

# Trying to use list as a key (will raise error)
try:
    invalid_dict = {[1, 2]: "List as key"}  # ❌
except TypeError as e:
    print("Error:", e)