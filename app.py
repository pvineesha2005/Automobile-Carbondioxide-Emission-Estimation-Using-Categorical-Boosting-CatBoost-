from flask import Flask, render_template, request, redirect, url_for, session
from catboost import CatBoostRegressor
import pandas as pd
import joblib

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load model and valid categories
model = CatBoostRegressor()
model.load_model("catboost_co2_model.cbm")
valid_categories = joblib.load("valid_categories.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form_data = {k: request.form[k] for k in request.form}

        try:
            input_data = {
                'MAKER': form_data["maker"].strip().upper(),
                'MODEL': form_data["model"].strip().upper(),
                'VEHICLECLASS': form_data["vehicle_class"].strip().upper(),
                'ENGINESIZE': float(form_data["engine_size"]),
                'CYLINDERS': int(form_data["cylinders"]),
                'TRANSMISSION': form_data["transmission"].strip().upper(),
                'FUEL': form_data["fuel"].strip().upper(),
                'FUELCONSUMPTION': float(form_data["fuel_consumption"])
            }


            fuel_map = {"PETROL": "X", "DIESEL": "Z"}
            input_data["FUEL"] = fuel_map.get(input_data["FUEL"], input_data["FUEL"])

            # Validate input categories
            for field in ['MAKER', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUEL']:
                if input_data[field] not in valid_categories.get(field, []):
                    session['warning'] = f"Invalid {field.lower().capitalize()}: '{input_data[field]}'"
                    session['form_data'] = form_data
                    return redirect(url_for('index'))

            # Prediction
            df = pd.DataFrame([input_data])
            prediction = round(model.predict(df)[0], 2)

            # Status interpretation
            if prediction <= 113:
                status = "Low CO2 Emission and environmental friendly."
            else:
                status = "High CO2 Emission "

            # Store for POST result
            session['prediction'] = prediction
            session['status'] = status
            session['form_data'] = form_data
            return redirect(url_for('index'))

        except Exception as e:
            session['warning'] = f"Error: {str(e)}"
            session['form_data'] = form_data
            return redirect(url_for('index'))


    prediction = session.pop('prediction', None)
    status = session.pop('status', None)
    warning = session.pop('warning', None)
    form_data = session.pop('form_data', {})

    return render_template("index.html", prediction=prediction, status=status, warning=warning, form_data=form_data)

if __name__ == "__main__":
    app.run(debug=True)
