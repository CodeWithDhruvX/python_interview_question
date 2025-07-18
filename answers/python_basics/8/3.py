def mood_response(user_input):
    user_input = user_input.lower()

    if "happy" in user_input:
        return "Awesome! Keep smiling 😊"
    elif "sad" in user_input:
        return "Aww, it's okay. Tomorrow will be better!"
    elif "angry" in user_input:
        return "Take a deep breath... Relax. It'll be fine."
    else:
        return "I'm here for you! Tell me more."

# Sample inputs
messages = [
    "I'm feeling really happy today!",
    "Ugh I'm so angry right now.",
    "Honestly, just a bit sad.",
    "Nothing much, just chilling."
]

for msg in messages:
    print(f"User: {msg}\nBot: {mood_response(msg)}\n")
