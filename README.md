# Mobile Price Classification & Prediction

An end-to-end machine learning project designed to classify mobile phones into price ranges based on their technical specifications. The solution includes exploratory data analysis, neural network modeling, and an interactive web interface built with Streamlit for real-time inference.

---

## Project Overview

The objective of this project is to predict the price category of a mobile device using various hardware and connectivity attributes (e.g., RAM, battery capacity, camera specifications, processor speed). The problem is framed as a multi-class classification task with four target categories:

- Low Cost (0)
- Medium Cost (1)
- High Cost (2)
- Very High Cost (3)

A deep learning model built with TensorFlow/Keras is trained on preprocessed tabular data, and inference is served via a Streamlit web application.

---

## Repository Structure

```text
.
├── LICENSE              # Open-source license terms
├── Mobile.csv           # Raw dataset containing mobile specifications
├── README.md            # Project documentation
├── app.py               # Streamlit application for deployment and inference
├── experiments.ipynb    # Jupyter Notebook for EDA, preprocessing, and model training
├── model.keras          # Trained deep learning model in native Keras 3 format
├── requirements.txt     # List of required Python dependencies
└── scaler.pkl           # Fitted feature scaler (StandardScaler/MinMaxScaler)
