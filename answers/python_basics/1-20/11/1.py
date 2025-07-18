def guess_the_number():
    secret_number = 7
    attempts = [3, 8, 5, 7, 2]
    
    for attempt in attempts:
        print(f"Trying: {attempt}")
        if attempt == secret_number:
            print("Correct guess! Breaking out of the loop.")
            break
        print("Wrong guess.")
    else:
        print("Secret number not found in the attempts.")

guess_the_number()
