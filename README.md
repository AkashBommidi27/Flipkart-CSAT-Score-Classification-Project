# Flipkart Customer Service Satisfaction Prediction

## Project Overview
This project focuses on predicting **Customer Satisfaction Scores (CSAT)** for Flipkart's customer support interactions using a supervised machine learning classification approach. The CSAT scores range from 1 (very dissatisfied) to 5 (very satisfied). By leveraging historical customer support data, the project aims to identify patterns that influence customer satisfaction, enabling proactive interventions to enhance service quality and customer retention.

The dataset, `Customer_support_data.csv`, contains detailed information about customer interactions, including ticket details, agent performance, response times, and customer feedback. The project employs advanced machine learning techniques, including **XGBoost**, to build a robust classification model, with SHAP analysis for interpretability.

### Objective
- Build a classification model to predict CSAT scores (1–5) based on customer support interaction features.
- Identify key factors influencing customer satisfaction to guide operational improvements.
- Provide actionable insights for Flipkart to enhance customer service workflows and reduce dissatisfaction.

### Dataset
The dataset (`Customer_support_data.csv`) includes the following key features:
- **Unique id**: Unique identifier for each support ticket.
- **channel_name**: Communication channel (e.g., Email, Chat, Phone).
- **category**: Broad issue category (e.g., Technical, Billing).
- **Sub-category**: Specific issue type (e.g., Login Failure).
- **Customer Remarks**: Free-text customer feedback (used for sentiment analysis).
- **Order_id**: Associated order ID.
- **order_date_time**: Timestamp of the order.
- **Issue_reported at**: Timestamp when the issue was reported.
- **issue_responded**: Timestamp of the agent's response.
- **Survey_response_Date**: Date of CSAT feedback submission.
- **Customer_City**: Customer's city.
- **Product_category**: Product type involved.
- **Item_price**: Price of the item.
- **connected_handling_time**: Time spent by the agent resolving the issue.
- **Agent_name**, **Supervisor**, **Manager**: Agent and team details.
- **Tenure Bucket**: Agent experience level (e.g., 0–6 months).
- **Agent Shift**: Shift during which the ticket was handled (e.g., Morning, Evening).
- **CSAT Score** (Target): Satisfaction score (1–5).

The dataset is available in the repository under `Customer_support_data.csv`.

## Project Structure
- `Flipkart_Classification_Project.ipynb`: Main Jupyter notebook containing the end-to-end machine learning pipeline, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and SHAP analysis.
- `Customer_support_data.csv`: Dataset used for training and testing the model.
- `README.md`: This file provides an overview and setup instructions.
- `requirements.txt`: List of Python dependencies required to run the project.

## Methodology
1. **Data Preprocessing**:
   - Handled missing values and encoded categorical variables using LabelEncoder.
   - Derived features such as `response_time_minutes` and `time_from_order_to_response`.
   - Standardized numerical features using StandardScaler.
2. **Exploratory Data Analysis (EDA)**:
   - Conducted Univariate, Bivariate, and Multivariate analyses with 15+ visualizations to uncover patterns in CSAT scores, response times, agent performance, and more.
   - Key insights include the impact of response time and agent tenure on satisfaction.
3. **Model Development**:
   - Evaluated multiple algorithms, with **XGBoost Classifier** achieving the best performance (Weighted F1 Score: 0.81, ROC AUC: 0.8920).
   - Applied cross-validation and hyperparameter tuning using GridSearchCV to optimize model performance.
4. **Model Interpretability**:
   - Used SHAP (SHapley Additive exPlanations) to identify key features influencing predictions, such as `chance_to_churn`, `sentiment_score`, `response_time_minutes`, and `connected_handling_time`.
5. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1 Score, and ROC AUC were used to assess model performance.
   - The model excels at identifying dissatisfied customers, enabling proactive interventions.

## Installation
To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AkashBommidi27/Flipkart-Customer-Satisfaction-ML.git
   cd Flipkart-Customer-Satisfaction-ML
