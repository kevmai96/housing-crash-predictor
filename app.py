import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

print("Generating synthetic economic dataset...")
numHouses = 500

data = pd.DataFrame({
    "interest_rate": np.random.uniform(1, 10, numHouses),
    "inflation_rate": np.random.uniform(1, 12, numHouses),
    "unemployment_rate": np.random.uniform(2, 12, numHouses),
    "housing_supply_index": np.random.uniform(50, 200, numHouses),
    "price_growth": np.random.uniform(-5, 20, numHouses),
    "gdp_growth": np.random.uniform(-3, 6, numHouses),
})

# Synthetic crash logic
risk_score = (
    data["interest_rate"] * 0.3 +
    data["inflation_rate"] * 0.2 +
    data["unemployment_rate"] * 0.3 +
    data["housing_supply_index"] * 0.01 -
    data["gdp_growth"] * 0.4 -
    data["price_growth"] * 0.1
)

# Crash if score exceeds threshold
data["crash_risk"] = (risk_score > 6).astype(int)

print(f"Dataset created with {len(data)} economic scenarios")

X = data.drop("crash_risk", axis=1)
y = data["crash_risk"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)
preds = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, preds)

print("\nModel performance")
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification report: ", classification_report(y_test, preds))
print("\nPredict housing crash risk")
print("Press enter to use default values\n")

try:
    interest = float(input("Interest rate (%) [5]: ") or 5)
    inflation = float(input("Inflation (%) [4]: ") or 4)
    unemployment = float(input("Unemployment (%) [5]: ") or 5)
    supply = float(input("Housing supply index [120]: ") or 120)
    price_growth = float(input("House price growth (%) [8]: ") or 8)
    gdp = float(input("GDP growth (%) [2]: ") or 2)

    new_data = pd.DataFrame([[
        interest,
        inflation,
        unemployment,
        supply,
        price_growth,
        gdp
    ]], columns=X.columns)

    new_scaled = scaler.transform(new_data)

    prediction = model.predict(new_scaled)[0]
    probability = model.predict_proba(new_scaled)[0][1]

    print("\nResult")

    if prediction == 1:
        print(f"HIGH CRASH RISK ({probability*100:.1f}% probability)")
    else:
        print(f"Low crash risk ({probability*100:.1f}% probability)")

except (ValueError, KeyboardInterrupt):
    print("\nPrediction skipped.")

print("\nDone.")
