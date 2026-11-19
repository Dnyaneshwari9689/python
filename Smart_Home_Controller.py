
# Intelligent Smart Home Controller

print("SMART HOME CONTROLLER")

temperature = float(input("Enter room temperature: "))
motion = input("Is motion detected? (yes/no): ").lower()

# Agent decision making

if motion == "yes":
    print("Light: ON")
else:
    print("Light: OFF")

if temperature > 30:
    print("Fan: ON")
else:
    print("Fan: OFF")

if temperature > 40:
    print("Warning: Room temperature is too high!")

print("Smart Home Control Completed.")
