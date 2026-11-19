
# Intelligent Chatbot Agent

print("Welcome to Intelligent Chatbot!")
print("Type 'bye' to exit.")

while True:
    user = input("\nYou: ").lower()

    # Agent decision making
    if "hello" in user or "hi" in user:
        print("Bot: Hello! How can I help you?")

    elif "name" in user:
        print("Bot: I am an Intelligent Chatbot.")

    elif "how are you" in user:
        print("Bot: I am fine. Thank you!")

    elif "weather" in user:
        print("Bot: Please check your weather application.")

    elif "help" in user:
        print("Bot: I can answer simple questions.")

    elif "bye" in user:
        print("Bot: Goodbye! Have a nice day.")
        break

    else:
        print("Bot: Sorry, I don't understand.")
