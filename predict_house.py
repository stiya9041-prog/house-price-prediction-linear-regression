import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model/house_price_model.pkl")

print("=" * 45)
print("HOUSE PRICE PREDICTION")
print("=" * 45)

def get_integer(prompt, minimum, maximum):
    while True:
        try:
            value = int(input(prompt))
            if minimum <= value <= maximum:
                return value
            print(f"Enter a value between {minimum} and {maximum}.")
        except ValueError:
            print("Please enter a whole number.")

def get_yes_no(prompt):
    while True:
        value = input(prompt).strip().lower()
        if value in ("yes", "no"):
            return value
        print("Please enter yes or no.")

# Collect and validate property details
area = get_integer("Enter area in square feet (1650-16200): ", 1650, 16200)
bedrooms = get_integer("Enter number of bedrooms (1-6): ", 1, 6)
bathrooms = get_integer("Enter number of bathrooms (1-4): ", 1, 4)
stories = get_integer("Enter number of stories (1-4): ", 1, 4)

mainroad = get_yes_no("Is it on the main road? (yes/no): ")
guestroom = get_yes_no("Does it have a guestroom? (yes/no): ")
basement = get_yes_no("Does it have a basement? (yes/no): ")
hotwaterheating = get_yes_no("Hot water heating? (yes/no): ")
airconditioning = get_yes_no("Air conditioning? (yes/no): ")

parking = get_integer("Number of parking spaces (0-3): ", 0, 3)
prefarea = get_yes_no("Is it in a preferred area? (yes/no): ")

valid_furnishing = ["furnished", "semi-furnished", "unfurnished"]

while True:
    furnishingstatus = input(
        "Furnishing status (furnished/semi-furnished/unfurnished): "
    ).strip().lower()

    if furnishingstatus in valid_furnishing:
        break

    print("Please select one of the listed furnishing statuses.")

# Prepare input in the format expected by the model
house = pd.DataFrame([{
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus
}])

# Predict price
predicted_price = model.predict(house)[0]

print("\n" + "=" * 45)
print(f"Estimated house price: ₹{predicted_price:,.0f}")
print("=" * 45)
print("Note: This is a model estimate, not a professional valuation.")
