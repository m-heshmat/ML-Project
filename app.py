from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Collect form data
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race/ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            # Convert input data to DataFrame
            input_df = data.get_data_as_data_frame()

            # Predict using the model pipeline
            pipeline = PredictPipeline()
            predicted_math_score = pipeline.predict(input_df)

            # Return the template with the prediction
            return render_template('home.html', prediction=predicted_math_score[0])

        except Exception as e:
            # Log the error and return the form with an error message
            print(f"Error occurred: {e}")
            return render_template('home.html', prediction="Error in prediction. Please try again.")


# Ensure the app runs
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
