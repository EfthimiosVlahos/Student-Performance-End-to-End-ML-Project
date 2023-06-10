## End to End Machine Learning Project:Student Performance: Efthimios Vlahos

### Introduction About the Data :

**The dataset** The goal is to predict math scores of students

There are 6 independent variables (including `id`):

* gender : sex of students -> (Male/female)
* race/ethnicity : ethnicity of students -> (Group A, B,C, D,E)
* parental level of education : parents' final education ->(bachelor's degree,some college,master's degree,associate's degree,high school)
* lunch : having lunch before test (standard or free/reduced)
* test preparation course : complete or not complete before test
* math score
* writing score

Target variable:
* math score

Dataset Source Link :
[https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv](https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv)

# Approach for the project 

1. Data Ingestion : 
    * In Data Ingestion phase the data is first read as csv. 
    * Then the data is split into training and testing and saved as csv file.

2. Data Transformation : 
    * In this phase a ColumnTransformer Pipeline is created.
    * for Numeric Variables first SimpleImputer is applied with strategy median , then Standard Scaling is performed on numeric data.
    * for Categorical Variables SimpleImputer is applied with most frequent strategy, then ordinal encoding performed , after this data is scaled with Standard Scaler.
    * This preprocessor is saved as pickle file.

3. Model Training : 
    * In this phase base model is tested . The best model found was Ridge regressor.

4. Prediction Pipeline : 
    * This pipeline converts given data into dataframe and has various functions to load pickle files and predict the final results in python.

5. Flask App creation : 
    * Flask app is created with User Interface to predict the gemstone prices inside a Web Application.

# Exploratory Data Analysis Notebook

Link : [EDA Notebook](https://github.com/EfthimiosVlahos/Student-Performance-End-to-End-ML-Project/blob/main/notebook/1%20.%20EDA%20STUDENT%20PERFORMANCE%20.ipynb)

# Model Training Approach Notebook

Link : [Model Training Notebook](https://github.com/EfthimiosVlahos/Student-Performance-End-to-End-ML-Project/blob/main/notebook/2.%20MODEL%20TRAINING.ipynb)
