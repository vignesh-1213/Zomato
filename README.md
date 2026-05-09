# Zomato Dataset Analysis

## Project Overview
This project is completed as an assignment for the Alfido Tech Data Science Internship. The goal is to analyze the Zomato Bangalore Restaurants dataset to extract insights on ratings, cuisines, location preferences, and factors affecting ratings, and then applying a Machine Learning model to predict restaurant ratings.

## Dataset
**Source**: Kaggle (bhanupratapbiswas/zomato)

**Initial Observations**:
1. The dataset contains nearly 50k+ records of restaurant details from Bangalore.
2. Contains several text fields and missing data (notably in ratings and approximate cost). 
3. Contains features such as `rate`, `votes`, `location`, `rest_type`, `approx_cost(for two people)`, and `cuisines`.
4. Needs substantial cleaning for numeric conversion (e.g. "4.1/5" to float `4.1` and "1,200" to float `1200.0`).

## Deliverables provided in this project:
1. `zomato_analysis.ipynb`: A complete Jupyter Notebook containing data cleaning, exploratory data analysis with Matplotlib & Seaborn visualizations (Heatmaps, Wordclouds, Scatter plots), and a Random Forest Regressor ML model.
2. `report.pdf`: A presentation-ready PDF report summarizing Key Findings and 5 Platform Recommendations for Alfido Tech based on the analysis.

## Instructions to Run
1. Download `zomato.csv` from [Kaggle](https://www.kaggle.com/datasets/bhanupratapbiswas/zomato).
2. Place the `zomato.csv` file in the same directory (`c:/Zomato`).
3. Run the notebook using Jupyter Lab or Google Colab:
   ```bash
   pip install -r requirements.txt
   jupyter notebook zomato_analysis.ipynb
   ```
