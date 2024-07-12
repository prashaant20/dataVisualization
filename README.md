# DataVisualization

# Overview:

This project creates an interactive sales analysis dashboard using Python's Dash framework. It includes visualizations such as sales distribution by product, monthly sales trends, and a scatter plot of quantity vs sales with year-based coloring. The dashboard incorporates a machine learning model (linear regression) to predict sales based on quantity and price. Users can explore data dynamically through interactive graphs, and model evaluation metrics such as Mean Squared Error (MSE) are displayed. The project aims to provide actionable insights into sales performance and trends for different products.

# Sales Analysis Dashboard:

This repository contains code to set up an interactive sales analysis dashboard using Python's Dash framework. The dashboard includes visualizations of sales data, model evaluation metrics, and interactive features to explore sales trends based on different products.

# Features:
•	Sales by Product: Visualizes sales distribution across different products.  
•	Monthly Sales Trends: Displays trends in monthly sales for selected products.  
•	Quantity vs Sales Scatter Plot: Shows how quantity sold relates to sales, with year  

# Understanding the Code:

# Structure:
•	app.py: Main script to run the Dash application.  
o	Imports necessary libraries (pandas, numpy, plotly.express, sklearn).  
o	Generates sample sales data and performs feature engineering to extract month and year.  
o	Sets up a linear regression model to predict sales based on quantity and price.  
o	Initializes a Dash app (Dash) and defines its layout (html.Div, dcc.Graph).  
o	Includes callbacks (@app.callback) to update graphs based on user interactions.  

# Components:
•	Data Generation: Uses pandas to create a DataFrame (df) with sample sales data.  
•	Feature Engineering: Extracts month and year from the Date column.  
•	Machine Learning Model: Implements linear regression (sklearn.linear_model.LinearRegression) to predict sales.  
•	Dash App: Sets up interactive components (dcc.Graph) for visualizations and model evaluation.  
•	Callbacks: Define functions (@app.callback) to update graphs dynamically based on user selections.  

# Enhancements:
•	Replace sample data with real-world data sources or APIs.  
•	Experiment with different machine learning models or feature combinations.  
•	Improve visualizations with additional insights like trend lines or confidence intervals.  

# Dependencies:

To run the Sales Analysis Dashboard, ensure you have the following dependencies installed:

1. Python: Version 3.6 or higher
2. Python Packages:
    a). pandas: For data manipulation and analysis
    b). numpy: For numerical operations
    c). plotly: For interactive data visualization
    d). scikit-learn: For machine learning model (LinearRegression)
    e). dash: For creating interactive web-based dashboards

