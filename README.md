    ---
title: AI Concrete Engineering Suite
emoji: 🏗️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
---
# 🏗️ AI Concrete Engineering Suite

### *Bridging Civil Engineering and Artificial Intelligence for Construction 5.0*

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## 📖 Project Overview
Concrete behavior is inherently complex, non-linear, and variable. Traditional methods for mix design rely heavily on time-consuming empirical testing, while structural health monitoring often depends on subjective manual inspections.

This project introduces an **Integrated AI Toolkit** that addresses the full lifecycle of concrete engineering:

1.  **Pre-Construction (Design Phase):** Uses deep learning (ANN) to predict concrete compressive strength based on complex mix proportions, enabling data-driven optimization of materials.
2.  **Post-Construction (Maintenance Phase):** Uses computer vision (CNN) to automate the detection of structural cracks from site images, enhancing safety and reducing inspection time.

## 🎯 Key Objectives
* **Predictive Modeling:** Eliminate the wait time of standard 28-day crushing tests by accurately predicting strength using historical data.
* **Sustainable Design:** Facilitate the use of supplementary cementitious materials (Fly Ash, Slag) by modelling their impact on strength, supporting low-carbon concrete goals.
* **Automated NDT:** Provide a non-destructive testing (NDT) tool for rapid structural assessment using simple camera images.

## 🛠️ System Architecture

### 1. Mix Design Optimizer (Regression Model)
* **Algorithm:** Artificial Neural Network (ANN) with dense layers.
* **Input Features:** Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age.
* **Performance:** Achieved **R² Score of ≈ 0.91** on the standard UCI Concrete Compressive Strength dataset (1,030 samples).
* **Utility:** Allows engineers to simulate "What-if" scenarios to minimize cement usage without compromising strength.

### 2. Crack Detection System (Classification Model)
* **Algorithm:** Convolutional Neural Network (CNN) with max-pooling and dropout layers.
* **Data:** Trained on a binary dataset of cracked vs. uncracked concrete surfaces.
* **Performance:** High accuracy in distinguishing structural cracks from healthy surfaces.
* **Utility:** Early warning system for infrastructure maintenance.

## 💻 Tech Stack
* **Language:** Python 3.10+
* **Deep Learning:** TensorFlow / Keras
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn
* **Interface:** Gradio (Web-based interactive dashboard)

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/the-adityatomar/Concrete-Strength-Crack-Detection.git](https://github.com/the-adityatomar/Concrete-Strength-Crack-Detection.git)
```

### 2. Install dependecies
```bash
pip install -r requirements.txt
```
### 3. Launch the dashboard
```bash
python app.py
```
##  🏛️ Architecture
![Alt text for the image](Assets/Architecture.png)

##  📊 Dashboard
### Concrete stength prediction dashboard:
![Alt text for the image](Assets/Dashboard_Prediction.png)

### Concrete stength detection dashboard:
![Alt text for the image](Assets/Dashboard_Detection.png)





