from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model, encoders & final column order
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")

    # Read raw form values
    form_data = request.form.to_dict()

    ordinal_cols = encoders["ordinal_cols"]
    nominal_cols = encoders["nominal_cols"]
    numeric_cols = encoders["numeric_cols"]

    enc_ord = encoders["ordinal"]
    enc_oh = encoders["onehot"]

    # Prepare empty arrays
    ord_values = []
    num_values = []
    nom_values = []

    # 1. NUMERIC FEATURES (Direct use)
    for col in numeric_cols:
        num_values.append(float(form_data[col]))

    # 2. ORDINAL FEATURES (Use OrdinalEncoder)
    for col in ordinal_cols:
        ord_values.append(int(form_data[col]))

    ord_values = enc_ord.transform([ord_values])[0]

    # 3. NOMINAL FEATURES (Use OneHotEncoder)
    nom_row = [form_data[col] for col in nominal_cols]
    nom_encoded = enc_oh.transform([nom_row])[0]

    # Final feature vector
    final_input = np.concatenate([num_values, ord_values, nom_encoded])

    # Ensure shape matches model requirement
    final_input = final_input.reshape(1, -1)

    # Predict
    pred = model.predict(final_input)[0]

    result = "Employee will Leave" if pred == 1 else "Employee will Stay"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
