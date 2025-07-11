from sklearn.model_selection import train_test_split

# Sample dataset
X = [[i] for i in range(10)]  # Features: 0 to 9
y = [i % 2 for i in range(10)]  # Labels: 0 or 1

# Splitting dataset into train and test sets (75% train, 25% test by default)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Output the results
print("Train Features:", X_train)
print("Train Labels:", y_train)
print("Test Features:", X_test)
print("Test Labels:", y_test)
