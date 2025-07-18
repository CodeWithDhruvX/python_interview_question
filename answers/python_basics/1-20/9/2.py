def keep_asking():
    while True:
        user_input = input("Type 'exit' to quit: ")
        if user_input.lower() == 'exit':
            print("Exiting the loop.")
            break
        else:
            print(f"You typed: {user_input}")

if __name__ == "__main__":
    keep_asking()
