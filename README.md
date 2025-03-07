# Health Insurance Cost Prediction

A machine learning project that analyzes and predicts individual health insurance costs using demographic and lifestyle factors. This repository contains the code, data processing pipelines, model development, and a web application built with Flask that delivers real-time predictions.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Features](#project-features)
- [Methodology](#methodology)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Scope](#future-scope)
- [License](#license)

---

## Overview

This project focuses on the analysis and prediction of health insurance costs using historical data and various machine learning algorithms. By leveraging Python’s rich ecosystem—including libraries such as NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn—the project guides users through:

- **Exploratory Data Analysis (EDA):** Identifying key factors like age, BMI, and smoking status that influence insurance charges.
- **Preprocessing:** Cleaning and preparing data for robust analysis.
- **Modeling:** Training multiple regression models (Linear Regression, Support Vector Regression, Ridge Regression, and Random Forest Regression) and comparing their performance.
- **Deployment:** Integrating the best-performing model into a Flask web application for user-friendly, real-time insurance cost predictions.

The project was executed under the guidance of Dr. Manisha Saini and represents a collaborative effort from the project team.

---

## Dataset

The project uses the `insurance.csv` dataset, which includes essential attributes such as:

- **Age**
- **Gender**
- **BMI (Body Mass Index)**
- **Number of Children**
- **Smoking Status**
- **Region**
- **Insurance Charges**

The dataset is sourced from OSF.io and is publicly available. You can download it from the following link:  
[https://osf.io/7u5gy](https://osf.io/7u5gy)  

---

## Project Features

- **Data Exploration & Visualization:** Detailed analysis with plots such as bar charts, scatter plots, and heatmaps to reveal trends and correlations.
- **Preprocessing Pipeline:** Conversion of categorical variables to numerical values, handling of missing data, and feature scaling.
- **Multiple Machine Learning Models:** Implementation of Linear Regression, Support Vector Regression (SVR), Ridge Regression, and Random Forest Regression.
- **Model Evaluation:** Utilization of metrics like R-squared and RMSE to compare and select the best-performing model.
- **Web Application Deployment:** A Flask-based web interface that allows end-users to input parameters and receive real-time insurance cost predictions.

---

## Methodology

The project is structured into several key phases:

1. **Exploratory Data Analysis (EDA):**  
   - Visualizing relationships such as *Age vs. Charge*, *BMI vs. Charge*, *Smoker vs. Charge*, and *Region vs. Charge*.
   - Analyzing distributions, skewness, and kurtosis to understand data characteristics.

2. **Data Preprocessing & Feature Engineering:**  
   - Cleaning the dataset, handling missing values, and encoding categorical data.
   - Engineering new features to better capture the underlying patterns in the data.

3. **Model Development & Evaluation:**  
   - Training multiple regression models and evaluating their performance using cross-validation, R-squared scores, and RMSE.
   - Random Forest Regression emerged as the best model for this dataset, demonstrating superior accuracy.

4. **Deployment with Flask:**  
   - The selected predictive model is integrated into a Flask web application, enabling a user-friendly interface for real-time predictions.

The detailed methodology is elaborated in the attached project report.

---

## Installation & Setup

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Dependencies

Install the required libraries using pip. It is recommended to use a virtual environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/health-insurance-cost-prediction.git
cd health-insurance-cost-prediction

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

*The `requirements.txt` file includes libraries such as:*
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Flask
- Pickle (for model serialization)

---

## Usage

### Running the Analysis & Model Training

You can run the Jupyter notebooks or Python scripts included in the repository to:
- Perform data preprocessing and exploratory data analysis.
- Train and evaluate different regression models.

For example:
```bash
python train_model.py
```

### Starting the Flask Web Application

After training and saving the best model, you can start the web server by running:

```bash
python app.py
```

Then, open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to use the health insurance cost prediction tool.

---

## Results

- **Predictive Accuracy:** The machine learning models provided accurate predictions, with the Random Forest Regression model achieving the highest performance.
- **Key Insights:**  
  - Age, BMI, and smoking status are major determinants of insurance charges.
  - Other factors like gender and region play a lesser, yet still significant, role.
  
Detailed evaluation metrics and analysis can be found in the project report.

---

## Future Scope

- **Enhanced Modeling Techniques:** Explore ensemble methods and deep learning to further improve prediction accuracy.
- **Real-Time Data Integration:** Incorporate real-time data streams from IoT devices and electronic health records (EHRs) to refine predictions.
- **User Interface Improvements:** Enhance the web application with more interactive visualizations and features for better user engagement.
- **Collaboration with Healthcare Providers:** Integrate additional datasets and collaborate with industry experts for broader applications in healthcare cost management.
---

## License

This project is licensed under the [MIT License](LICENSE).

---

*For further details, please refer to the attached [project report](./Health_Insurance_Price_Prediction_Report.pdf).*

