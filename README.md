<H3>ENTER YOUR NAME</H3>DEEPAK RAJ S
<H3>ENTER YOUR REGISTER NO.</H3>212222240023
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
Developed by: DEEPAK RAJ S
RegisterNumber: 212222240023

import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#Reading the dataset
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df
#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
#Checking for null values
df.isnull().sum()
#Checking for duplicate values
df.duplicated()
#Describing the dataset
df.describe()
#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1
#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y
#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```
## OUTPUT:
DATASET:

![image](https://github.com/DEEPAK2200233/Ex-1-NN/assets/118707676/564dbb82-2f4f-4a86-b8d9-681f46ea1519)

DROPPING THE UNWANTED DATASET:

![image](https://github.com/DEEPAK2200233/Ex-1-NN/assets/118707676/9bb25efd-5226-46ea-aef1-de170cbf1f6c)

CHECKING NULL VALUES:

![image](https://github.com/DEEPAK2200233/Ex-1-NN/assets/118707676/91474fee-123d-4175-b258-92bbf75723cb)

CHECKING FOR DUPLICATION:

![image](https://github.com/DEEPAK2200233/Ex-1-NN/assets/118707676/90c71c87-f527-4e00-9c0d-3f36b6748e67)

DESCRIBING THE DATASET:

![image](https://github.com/DEEPAK2200233/Ex-1-NN/assets/118707676/02c414f1-451f-4baa-9298-afd1b4d1fedf)

SCALING THE DATASET:
![image](https://github.com/DEEPAK2200233/Ex-1-NN/assets/118707676/9da6e306-00ef-469d-94b0-162f0e51acdb)

X FEATURES:
![image](https://github.com/DEEPAK2200233/Ex-1-NN/assets/118707676/d86e8b16-6046-4b4b-ae39-d4321e70d102)

Y FEATURES:
![image](https://github.com/DEEPAK2200233/Ex-1-NN/assets/118707676/cb0e7656-c6c0-458d-b95d-d3111a07dbc4)

SPLITTING THE TRAINING AND TESTING DATASET:
![image](https://github.com/DEEPAK2200233/Ex-1-NN/assets/118707676/5f7fb6ae-6e05-4a1b-b4b5-201e51ccb22d)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


