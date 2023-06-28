# End to End Machine Learning Project: Student Performance: Efthimios Vlahos

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
- [01. Data Overview](#data-overview)
- [02. EDA:Sneak Peak](#data-EDA)
     - [Adding Columns](#data-Adding)
     - [Exploring Data ( Visualization )](data-exploration)
- [03. Modelling](#modelling-overview)
- [04. Modelling Summary](#modelling-summary)


# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

This project understands how the student's performance (test scores) is affected by other variables such as Gender, Ethnicity, Parental level of education, Lunch and Test preparation course.

### Actions <a name="overview-actions"></a>

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
  
## In Progress:

6. Project deployment In AWS Cloud Suing CICD Pipelines : 

7. Deployment of ML application in Azure cloud Using github Actions : 

As we are predicting a countinous output,tested a variety of models, namely:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Random Forest Regressor
* CatBoosting Regressor
* AdaBoost Regressor
* XGBRegresssor
* KNN
* Decision Tree


- For each model, we will import the data in the same way but will need to pre-process the data based up the requirements of each particular algorithm.  We will train & test each model, look to refine each to provide optimal performance, and then measure this predictive performance based on several metrics to give a well-rounded overview of which is best.

### Results: Sneak Peak <a name="overview-results"></a>

<img width="536" alt="Screenshot 2023-06-10 at 11 30 05 AM" src="https://github.com/EfthimiosVlahos/Student-Performance-End-to-End-ML-Project/assets/56899588/b226653b-8215-425a-a87a-1882dc353193">
___

# Data Overview  <a name="data-overview"></a>

- After this data pre-processing in Python, we have a dataset for modelling that contains the following fields...

| **Variable Name** | **Variable Type** | **Description** |
|---|---|---|
| Gender | Independent| Sex of students |
| race_ethnicity | Independent | Ethnicity of students |
| Parental_level_of_education | Independent | Parents final education level|
| lunch | Independent | Having lunch before test |
| test_preparation_course | Independent | Complete or not complete beofre test|
| writing_score| Independent | Score on written exams|
| math_score | Dependent | Score on math exams |

# EDA  <a name="data-EDA"></a>
- First, imported necessary libraries for EDA.
```python

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

```
- Then standard preliminary look at the data:
<img width="1079" alt="Screenshot 2023-06-26 at 7 13 39 PM" src="https://github.com/EfthimiosVlahos/Student-Performance-End-to-End-ML-Project/assets/56899588/8c3d778a-d579-4ddf-8f2f-0dbad6f5ee98">

## Data Checks to Perform:
* Check Missing values
* Check Duplicates
* Check data type
* Check the number of unique values of each column
* Check statistics of data set
* Check various categories present in the different categorical column

- In the EDA notebook, you will see that there were (thankfully) no missing values and no duplicates from the kaggle dataset. The data types were as follows:

<img width="1140" alt="Screenshot 2023-06-26 at 7 52 53 PM" src="https://github.com/EfthimiosVlahos/Student-Performance-End-to-End-ML-Project/assets/56899588/4f0e6562-0cd8-4905-83b7-c59601872f8f">

- As far as the datatypes and distribution of numerical columns of dataset, they were as follows:

<img width="1140" alt="Screenshot 2023-06-26 at 7 54 19 PM" src="https://github.com/EfthimiosVlahos/Student-Performance-End-to-End-ML-Project/assets/56899588/090ce5c5-7639-41be-a73e-925fd339756b">

- From above description of numerical data, all means are very close to each other - between 66 and 68.05;
- All standard deviations are also close - between 14.6 and 15.19;
- While there is a minimum score 0 for math, for writing minimum is much higher = 10 and for reading myet higher = 17



### Categorical variables
- There were a few catrgorical variables in the dataset, so decided to list all categories of each column:

```python
print("Categories in 'gender' variable:     ",end=" " )
print(df['gender'].unique())

print("Categories in 'race_ethnicity' variable:  ",end=" ")
print(df['race_ethnicity'].unique())

print("Categories in'parental level of education' variable:",end=" " )
print(df['parental_level_of_education'].unique())

print("Categories in 'lunch' variable:     ",end=" " )
print(df['lunch'].unique())

print("Categories in 'test preparation course' variable:     ",end=" " )
print(df['test_preparation_course'].unique())


#Output:
#Categories in 'gender' variable:      ['female' 'male']
#Categories in 'race_ethnicity' variable:   ['group B' 'group C' 'group A' 'group D' 'group E']
#Categories in'parental level of education' variable: ["bachelor's degree" 'some college' "master's degree" "associate's degree"
 'high school' 'some high school']
#Categories in 'lunch' variable:      ['standard' 'free/reduced']
#Categories in 'test preparation course' variable:      ['none' 'completed']
```
- To get a number on the amount of categorical variables and numerical variables, did the following:
```python
#define numerical & categorical columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))

#Output:
#We have 3 numerical features : ['math_score', 'reading_score', 'writing_score']
#We have 5 categorical features : ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
```
### Adding Columns <a name="data-Adding"></a>
- Decide to create two new columns, "total_score" and "average" which could potentially be better attributes to predict math scores. Here is the following code:
```python
df['total score'] = df['math_score'] + df['reading_score'] + df['writing_score']
df['average'] = df['total score']/3
df.head()
```
- Also wanted to see which students got a perfect scores on math, reading, and writing as well as number of students who underperformed:

```python
reading_full = df[df['reading_score'] == 100]['average'].count()
writing_full = df[df['writing_score'] == 100]['average'].count()
math_full = df[df['math_score'] == 100]['average'].count()

print(f'Number of students with full marks in Maths: {math_full}')
print(f'Number of students with full marks in Writing: {writing_full}')
print(f'Number of students with full marks in Reading: {reading_full}')

#Output
#Number of students with full marks in Maths: 7
#Number of students with full marks in Writing: 14
#Number of students with full marks in Reading: 17

reading_less_20 = df[df['reading_score'] <= 20]['average'].count()
writing_less_20 = df[df['writing_score'] <= 20]['average'].count()
math_less_20 = df[df['math_score'] <= 20]['average'].count()

print(f'Number of students with less than 20 marks in Maths: {math_less_20}')
print(f'Number of students with less than 20 marks in Writing: {writing_less_20}')
print(f'Number of students with less than 20 marks in Reading: {reading_less_20}')

#Output
#Number of students with less than 20 marks in Maths: 4
#Number of students with less than 20 marks in Writing: 3
#Number of students with less than 20 marks in Reading: 1
```
#### Insights
- From above values we get students have performed the worst in Maths
- Best performance is in reading section


## Data Exploration <a name="data-exploration"></a>
### Histogram and KDE

<img width="1140" alt="Screenshot 2023-06-27 at 4 59 08 PM" src="https://github.com/EfthimiosVlahos/Student-Performance-End-to-End-ML-Project/assets/56899588/ac74772c-e12a-4233-ab10-3e49fe825be9">

<img width="1140" alt="Screenshot 2023-06-27 at 4 59 39 PM" src="https://github.com/EfthimiosVlahos/Student-Performance-End-to-End-ML-Project/assets/56899588/333675c2-484d-4df4-85b6-5cd67d880856">

#### Inisghts
- Female students tend to perform well then male students.

### Parental level of Education
<img width="1140" alt="Screenshot 2023-06-27 at 5 01 01 PM" src="https://github.com/EfthimiosVlahos/Student-Performance-End-to-End-ML-Project/assets/56899588/220e47f0-93d4-4a6f-b4e6-3a303a8fccf2">

#### Insights
- In general parent's education don't help student perform well in exam.
- 2nd plot shows that parent's whose education is of associate's degree or master's degree their male child tend to perform well in exam
- 3rd plot we can see there is no effect of parent's education on female students.


### Gender Distribution
<img width="1140" alt="Screenshot 2023-06-27 at 5 04 45 PM" src="https://github.com/EfthimiosVlahos/Student-Performance-End-to-End-ML-Project/assets/56899588/8098ef96-154c-4c93-a6f4-0039ba630647">

#### Insights
- Gender has balanced data with female students are 518 (48%) and male students are 482 (52%)

### Race/Ethnicity
<img width="1140" alt="Screenshot 2023-06-27 at 5 06 35 PM" src="https://github.com/EfthimiosVlahos/Student-Performance-End-to-End-ML-Project/assets/56899588/7ec9480a-abd3-459e-a637-7846864da410">


##### Insights
- Most of the student belonging from group C /group D.
- Lowest number of students belong to groupA.

### Outliers
<img width="1140" alt="Screenshot 2023-06-27 at 5 08 04 PM" src="https://github.com/EfthimiosVlahos/Student-Performance-End-to-End-ML-Project/assets/56899588/91c99f57-011e-416e-91e1-b9a128dedbe6">

- Dont seem to be any outliers in the data via boxplot

### Linearity with Independent Variables
- Using a simple pairplot, we are able to see if the independent variables are linearity correlated to the dependednt variable:
```python
sns.pairplot(df,hue = 'gender')
plt.show()
```
### Conclusions
- Student's Performance is related with lunch, race, parental level education
- Females lead in pass percentage and also are top-scorers
- Student's Performance is not much related with test preparation course
- Finishing preparation course is benefitial.





# Modelling  <a name="modelling-overview"></a>

## Data Imports
- First, imported all algorithms and metrics to be used in the project
```python
# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings

````

### Seperating the independent (predictor) and dependent (target) variables

```python 
X = df.drop(columns=['math_score'],axis=1)
y = df['math_score']
```

### Data Preprocessing

```python
# Create Column Transformer with 3 types of transformers
num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, cat_features),
         ("StandardScaler", numeric_transformer, num_features),        
    ]
)
X = preprocessor.fit_transform(X)

```
### Seperate Data into training and testing sets

```python
# separate dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape, X_test.shape

```
### Function to display results on training and testing set for algorithms
```python
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(), 
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor()
}
model_list = []
r2_list =[]

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate Train and Test dataset
    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)

    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    
    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))

    print('----------------------------------')
    
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    r2_list.append(model_test_r2)
    
    print('='*35)
    print('\n')

```

# Modelling Summary  <a name="modelling-summary"></a>

The goal for the project was to build a model that would accurately predict students math score. This would allow for a much more targeted approach when helping students obtain a better math score.

Here we see which models performed best with respect to three metrics on the test set:

**Metric 1: Root Mean Squared Error on Test set**

* Linear Regression = 5.35
* Lasso Regression = 6.52
* Ridge Regression = 5.39
* K-Neighbors Regressor = 7.25
* Decision Tree Regressor=7.63
* Random Forest Regressor= 6.01
* XGBRegressor=6.59
* CatBoosting Regressor=6.00
* AdaBoost Regressor=6.01

**Metric 2: Mean Absolute Error on Test set**

* Linear Regression = 4.21
* Lasso Regression = 5.16
* Ridge Regression = 4.21
* K-Neighbors Regressor = 5.62
* Decision Tree Regressor=6.02
* Random Forest Regressor=4.72
* XGBRegressor=5.08
* CatBoosting Regressor=4.61
* AdaBoost Regressor=4.68

**Metric 3: R2 Score on Test set**

* Linear Regression = .88
* Lasso Regression = .83
* Ridge Regression = .88
* K-Neighbors Regressor = .76
* Decision Tree Regressor= .76
* Random Forest Regressor= .85
* XGBRegressor= .82
* CatBoosting Regressor= .85
* AdaBoost Regressor= .85





