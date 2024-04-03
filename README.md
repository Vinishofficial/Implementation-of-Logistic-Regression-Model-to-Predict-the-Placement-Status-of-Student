# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
  1. Import the standard libraries.
  2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
  3. Import LabelEncoder and encode the dataset.
  4. Import LogisticRegression from sklearn and apply the model on the dataset.
  5. Predict the values of array.
  6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
  7. Apply new unknown values
## Program:
 ```
 Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
 Developed by: VINISH RAJ R
 RegisterNumber: 212223230243
```
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### Placement_data:
![AIML4 1](https://github.com/Rajeshanbu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118924713/54a62258-b91f-4aac-b0ee-2009d97046f1)

### Salary_data:
![AIML4 2](https://github.com/Rajeshanbu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118924713/4abf3c8d-be23-482d-8202-8d3c10bc3741)

### ISNULL():
![AIML4 3](https://github.com/Rajeshanbu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118924713/87d4379b-23ed-42b4-909e-70e55fb0d120)


### DUPLICATED():
![AIML4 4](https://github.com/Rajeshanbu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118924713/9bfd5f56-a1b1-48a1-8eee-8bced3a4377f)

### Print Data:
![AIML4 5](https://github.com/Rajeshanbu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118924713/df436c7b-c0db-4ff7-9848-d16c64aef93b)

### iloc[:,:-1]:
![AIML4 6](https://github.com/Rajeshanbu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118924713/04404df1-ae6f-4bd8-8457-3318655ba91e)

### Data_Status:
![AIML4 7](https://github.com/Rajeshanbu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118924713/eaf8b031-5eae-46bd-b670-e9e967228ca6)

### Y_Prediction array:
![AIML4 8](https://github.com/Rajeshanbu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118924713/f2eeca58-481e-44eb-af93-c11303f2aa82)

### Accuray value:
![AIML4 9](https://github.com/Rajeshanbu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118924713/daadec14-9939-40e8-ba05-c57dcf27469a)

### Confusion Array:
![AIML 4 10](https://github.com/Rajeshanbu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118924713/ade73f46-9569-4c6d-91ff-92714538b240)

### Classification report:
![Screenshot 2024-03-18 111833](https://github.com/Rajeshanbu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118924713/19851979-eed2-4743-9ad7-aea964ee9042)
### Prediction of LR:
![Screenshot 2024-03-18 111935](https://github.com/Rajeshanbu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118924713/dbc871f4-bbce-40d4-ba57-6cda4f3a677b)
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
