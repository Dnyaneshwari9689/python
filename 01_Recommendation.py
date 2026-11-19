
# Intelligent Recommendation Assistant

print("MOVIE RECOMMENDATION SYSTEM")

print("\nAvailable Genres:")
print("1. Action")
print("2. Comedy")
print("3. Horror")
print("4. Romance")
print("5. Science Fiction")

choice = input("\nEnter your favourite genre: ").lower()

# Agent decision making

if choice == "1":
    print("Recommended Movies:")
    print("1. Avengers")
    print("2. Mission Impossible")

elif choice == "2":
    print("Recommended Movies:")
    print("1. Home Alone")
    print("2. The Mask")

elif choice == "3":
    print("Recommended Movies:")
    print("1. The Conjuring")
    print("2. Insidious")

elif choice == "4":
    print("Recommended Movies:")
    print("1. Titanic")
    print("2. The Notebook")

elif choice == "5":
    print("Recommended Movies:")
    print("1. Interstellar")
    print("2. The Martian")

else:
    print("Sorry! Genre not available.")

print("\nThank you for using the Recommendation Assistant!")
