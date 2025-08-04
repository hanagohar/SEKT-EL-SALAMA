from datetime import datetime
import pandas as pd
import joblib

# Load model and preprocessor
preprocessor = joblib.load(r"D:\Desktop\DS\ds\3-machine learning\preprocessor.pkl")
model = joblib.load(r"D:\Desktop\DS\ds\3-machine learning\model.pkl")

year = int(input("Enter year (e.g. 2024): "))
month = int(input("Enter month (e.g. 3): "))
day = int(input("Enter day (e.g. 1): "))
hour = int(input("Enter hour (e.g. 7): "))
minute = int(input("Enter minute (e.g. 30): "))
second = int(input("Enter second (e.g. 0): "))


def print_input_guide(title, unit, min_val, max_val, levels):
    print(f"\n{title} ({unit}) — Range: {min_val} to {max_val}")
    print("-" * 50)
    for level in levels:
        print(f" {level[0]:<10} → {level[1]}")
    print("-" * 50)


# 🚗 Vehicle Count
print_input_guide(
    "🚗 Vehicle Count", "0–300", 0, 300,
    [("0–50", "Very Low traffic"),
     ("51–100", "Low traffic"),
     ("101–200", "Moderate traffic"),
     ("201–300", "Heavy traffic")]
)
vehicle_count = int(input("Enter vehicle count: "))

# ⚡ Traffic Speed
print_input_guide(
    "⚡ Traffic Speed", "km/h", 0, 160,
    [("0–20", "Congested"),
     ("21–50", "Slow traffic"),
     ("51–100", "Normal speed"),
     ("101–160", "High speed / Highway")]
)
speed = float(input("Enter traffic speed: "))

# 🚦 Road Occupancy
print_input_guide(
    "🚦 Road Occupancy", "%", 0, 100,
    [("0–30%", "Low occupancy (free flow)"),
     ("31–60%", "Moderate occupancy"),
     ("61–100%", "High occupancy (congested)")]
)
occupancy = float(input("Enter road occupancy: "))

# 🅿️ Parking Availability
print_input_guide(
    "🅿️ Parking Availability", "0–100", 0, 100,
    [("0–30", "Limited parking"),
     ("31–70", "Moderate availability"),
     ("71–100", "Plenty of parking")]
)
parking = int(input("Enter parking availability: "))


# Other inputs
traffic_light = input("\n🚥 Enter traffic light state (Red / Yellow / Green): ")
weather = input("🌦️Enter weather condition (Clear / Rain / Fog / Snow): ")
traffic_condition = input("📈 Enter traffic condition (Low / Medium / High): ")

# Time features
timestamp = datetime(year, month, day, hour, minute, second)
hour = timestamp.hour
month = timestamp.month
dayofweek = timestamp.weekday()
is_weekend = 1 if dayofweek >= 5 else 0
is_rush_hour = 1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0

# Prepare input DataFrame
input_data = pd.DataFrame([{
    "Vehicle_Count": vehicle_count,
    "Traffic_Speed_kmh": speed,
    "Road_Occupancy_%": occupancy,
    "Traffic_Light_State": traffic_light,
    "Weather_Condition": weather,
    "Parking_Availability": parking,
    "Traffic_Condition": traffic_condition,
    "Hour": hour,
    "Month": month,
    "DayOfWeek": dayofweek,
    "IsWeekend": is_weekend,
    "IsRushHour": is_rush_hour,
    "Speed_Occupancy_Interaction": speed * occupancy,
    "Vehicle_Weather_Interaction": vehicle_count * (["Clear", "Rain", "Fog", "Snow"].index(weather) + 1),

    # Fixed values
    "Sentiment_Score": 0.0,
    "Ride_Sharing_Demand": 20,
    "Emission_Levels_g_km": 300,
    "Energy_Consumption_L_h": 15
}])

# Transform and predict
X_transformed = preprocessor.transform(input_data)
prediction = model.predict(X_transformed)

# Result

print("\n🚦 Accident Prediction:", "⚠️ Accident Likely" if prediction[0] >= 0.2 else "✅ No Accident Expected")

print(f"Prediction Value: {prediction[0]}")
