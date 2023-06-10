# End to End Machine Learning Project:Student Performance: Efthimios Vlahos

## Introduction About the Data :

#### The dataset ####

There are 6 independent variables:

* 'gender' : sex of students -> (Male/female)
* 'race_ethnicity' : ethnicity of students -> (Group A, B,C, D,E)
* 'parental_level_of_education' : parents' final education ->(bachelor's degree,some college,master's degree,associate's degree,high school)
* 'lunch' : having lunch before test (standard or free/reduced)
* 'test_preparation_course' : complete or not complete before test
* 'writing score': score on written exams

Target variable:
* 'math score': score on math exams

Dataset Source Link :
[https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977)

# Approach for the project 

1. Data Ingestion : 
    * Imported and prepared data from various sources to be used for project, ensuring that it is in a format suitable for analysis and modeling.

2. Data Transformation : 
    * ColumnTransformer Pipeline was created.
    * For Numeric Variables first SimpleImputer is applied with strategy median , then Standard Scaling is performed on numeric data.
    * For Categorical Variables SimpleImputer is applied with most frequent strategy, then ordinal encoding performed , after this data is scaled with Standard Scaler.
    * Preprocessor is saved as pickle file.

3. Model Training : 
    * Various models are tested . The model that performed the best on the test set, with respect to R^2 score, was the Ridge regressor.
 
4. Flask App creation : 
    * Created Prediction pipeline using Flask Web App to predict math scores.
# In Progress

6. Project deployment In AWS Cloud Suing CICD Pipelines : 

7. Deployment of ML application in Azure cloud Using github Actions : 
 


# Exploratory Data Analysis Notebook

Link : [EDA Notebook](https://github.com/EfthimiosVlahos/Student-Performance-End-to-End-ML-Project/blob/main/notebook/1%20.%20EDA%20STUDENT%20PERFORMANCE%20.ipynb)

## Sneak Peak:

![image](<img width="1034" alt="Screenshot 2023-06-10 at 11 11 47 AM" src="https://github.com/EfthimiosVlahos/Student-Performance-End-to-End-ML-Project/assets/56899588/72a8f14d-0019-4441-9f38-fa4f0672abe3">)


# Model Training Approach Notebook

Link : [Model Training Notebook](https://github.com/EfthimiosVlahos/Student-Performance-End-to-End-ML-Project/blob/main/notebook/2.%20MODEL%20TRAINING.ipynb)
