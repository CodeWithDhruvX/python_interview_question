def interactive_type_checker():
    print("🔍 Python Type Detector (Type 'exit' to quit)")
    while True:
        user_input = input("Enter a Python expression: ")
        if user_input.strip().lower() == "exit":
            print("Goodbye! 👋")
            break
        try:
            evaluated = eval(user_input)
            print(f"✅ Value: {evaluated!r} | Type: {type(evaluated).__name__}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_type_checker()
