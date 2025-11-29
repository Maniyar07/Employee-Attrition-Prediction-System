import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# ===========================================
# 1. LOAD DATASET
# ===========================================
df = pd.read_csv("employee_attrition_dataset.csv")

# Remove ID
df = df.drop(columns=["Employee_ID"])

# Target
y = df["Attrition"]
X = df.drop("Attrition", axis=1)

# ===========================================
# 2. DEFINE COLUMN TYPES
# ===========================================
ordinal_cols = [
    "Work_Life_Balance",
    "Job_Satisfaction",
    "Work_Environment_Satisfaction",
    "Relationship_with_Manager",
    "Job_Involvement",
    "Performance_Rating",
    "Job_Level"
]

nominal_cols = [
    "Gender",
    "Marital_Status",
    "Department",
    "Job_Role",
    "Overtime"
]

numeric_cols = [
    col for col in X.columns
    if col not in ordinal_cols + nominal_cols
]

# ===========================================
# 3. ENCODERS
# ===========================================
enc_ord = OrdinalEncoder()
enc_oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

# Fit encoders
X[ordinal_cols] = enc_ord.fit_transform(X[ordinal_cols])

onehot_data = enc_oh.fit_transform(X[nominal_cols])
onehot_df = pd.DataFrame(onehot_data, columns=enc_oh.get_feature_names_out(nominal_cols))

# Final training table
X_final = pd.concat([X[numeric_cols + ordinal_cols], onehot_df], axis=1)

# Save the final column names
columns = X_final.columns.tolist()
pickle.dump(columns, open("columns.pkl", "wb"))

# Save encoders for Flask
encoders = {
    "ordinal": enc_ord,
    "onehot": enc_oh,
    "ordinal_cols": ordinal_cols,
    "nominal_cols": nominal_cols,
    "numeric_cols": numeric_cols
}
pickle.dump(encoders, open("encoders.pkl", "wb"))

# ===========================================
# 4. TRAIN MODEL
# ===========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model, Encoders & Columns Saved Successfully!")
