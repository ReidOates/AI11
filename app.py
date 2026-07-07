from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

model = load_model(
    "gtsrb_vgg16.keras"
)

class_names = {
    0:"Speed Limit 20 km/h",
    1:"Speed Limit 30 km/h",
    2:"Speed Limit 50 km/h",
    3:"Speed Limit 60 km/h",
    4:"Speed Limit 70 km/h",
    5:"Speed Limit 80 km/h",
    6:"End Speed Limit 80 km/h",
    7:"Speed Limit 100 km/h",
    8:"Speed Limit 120 km/h",
    9:"No Passing",
    10:"No Passing Trucks",
    11:"Right-of-Way",
    12:"Priority Road",
    13:"Yield",
    14:"Stop",
    15:"No Vehicles",
    16:"Truck Prohibited",
    17:"No Entry",
    18:"Danger",
    19:"Curve Left",
    20:"Curve Right",
    21:"Double Curve",
    22:"Bumpy Road",
    23:"Slippery Road",
    24:"Road Narrows",
    25:"Road Work",
    26:"Traffic Signal",
    27:"Pedestrian",
    28:"Children Crossing",
    29:"Bicycle Crossing",
    30:"Snow",
    31:"Wild Animals",
    32:"End Restriction",
    33:"Turn Right",
    34:"Turn Left",
    35:"Ahead Only",
    36:"Straight or Right",
    37:"Straight or Left",
    38:"Keep Right",
    39:"Keep Left",
    40:"Roundabout",
    41:"End No Passing",
    42:"End No Passing Trucks"
}


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    save_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(save_path)

    image = cv2.imread(save_path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (48,48)
    )

    image = image.astype(np.float32)

    image = preprocess_input(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        image,
        verbose=0
    )

    pred = np.argmax(
        prediction
    )

    confidence = float(
        np.max(prediction) * 100
    )

    return render_template(
        "result.html",
        prediction=class_names[pred],
        confidence=round(confidence,2),
        image_path=save_path
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=7860
    )