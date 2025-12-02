import numpy as np
import pandas as pd
from flask import Flask, request, render_template

from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("home.html")
    else:
        gender = request.form.get("gender")
        race_ethnicity = request.form.get("race_ethnicity")
        parental_level_of_education = request.form.get("parental_level_of_education")
        lunch = request.form.get("lunch")
        test_preparation_course = request.form.get("test_preparation_course")
        reading_score = int(request.form.get("reading_score"))
        writing_score = int(request.form.get("writing_score"))

        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score,
        ).get_data_as_data_frame()

        pred_df = PredictionPipeline().make_predictions(data)
        result = pred_df[0]
        return render_template("home.html", results=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)