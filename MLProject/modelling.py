import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data preprocessed
df = pd.read_csv("students_clean.csv")
X = df.drop("MathPass", axis=1)
y = df["MathPass"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Latih model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluasi sederhana (opsional)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Akurasi model di CI: {acc:.4f}")