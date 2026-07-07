# 🚦 Traffic Sign Recognition using Transfer Learning VGG16

A web-based image classification system for recognizing German Traffic Signs (GTSRB) using **Transfer Learning VGG16** with **TensorFlow/Keras** and **Flask**.

The application allows users to upload an image of a traffic sign and predicts its class along with the confidence score.

---

## Features

- Transfer Learning using VGG16
- German Traffic Sign Recognition Benchmark (GTSRB)
- Image upload through web interface
- Prediction confidence
- Responsive Flask web application
- Ready for Docker & Hugging Face Spaces deployment

---

## Technologies

- Python
- TensorFlow / Keras
- Flask
- OpenCV
- NumPy
- HTML
- CSS

---

## Dataset

German Traffic Sign Recognition Benchmark (GTSRB)

Dataset:
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

The dataset contains **43 traffic sign classes** used for training and evaluation.

---

## Project Structure

```
AI11/
│
├── app.py
├── gtsrb_vgg16.keras
├── requirements.txt
├── Dockerfile
│
├── static/
│   ├── style.css
│   └── uploads/
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── notebook.ipynb
```

---

## Installation

Clone repository

```
git clone https://github.com/ReidOates/AI11.git
```

Create virtual environment

```
python -m venv venv
```

Activate virtual environment

Windows

```
venv\Scripts\activate
```

Linux / MacOS

```
source venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

Run application

```
python app.py
```

Open browser

```
http://127.0.0.1:7860
```

---

## Model

Architecture

- Transfer Learning VGG16
- Image Size : 48 × 48
- Optimizer : Adam
- Loss : Sparse Categorical Crossentropy
- Output : 43 Classes

---

## Deployment

GitHub

https://github.com/ReidOates/AI11

Hugging Face Spaces

(Add your deployment URL here)

---

## Author

Muhammad Emil Mushthopa

Universitas Bale Bandung

Informatics Engineering
