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
![Screenshot 2024-04-03 110427](https://github.com/Vinishofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146931793/31b97185-e77e-48ef-86b2-b2ea2b17db89)


### Salary_data:
![Screenshot 2024-04-03 110427](https://github.com/Vinishofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146931793/cff1e799-fdcb-40a1-b8c3-d8859990a18f)


### ISNULL():
![Screenshot 2024-04-03 110436](https://github.com/Vinishofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146931793/55043109-6af5-4718-8614-e86bf1f7c42c)



### DUPLICATED():
![Screenshot 2024-04-03 110444](https://github.com/Vinishofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146931793/7bf392b1-d8e5-424d-9a01-96eaec5d5bfc)

### Print Data:
![Screenshot 2024-04-03 110456](https://github.com/Vinishofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146931793/b28806c1-2bb0-4cda-848e-836b0fb443c0)


### iloc[:,:-1]:
![Screenshot 2024-04-03 110456](https://github.com/Vinishofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146931793/3a2a742a-6069-41f3-876b-2107305268e8)

### Data_Status:
![Screenshot 2024-04-03 110509](https://github.com/Vinishofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146931793/13cd4c36-4f8c-41f6-b163-b1fe6b07742c)


### Y_Prediction array:
![Screenshot 2024-04-03 110516](https://github.com/Vinishofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146931793/9beda7c6-7302-4416-80c6-aa361e35abac)

### Accuray value:
![Screenshot 2024-04-03 110522](https://github.com/Vinishofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146931793/89e550c2-ab3c-446e-8296-359f010e0971)


### Confusion Array:
![Screenshot 2024-04-03 110527](https://github.com/Vinishofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146931793/3eaf436c-1269-48cd-8a9e-0e9e876468f2)


### Classification report:
![Screenshot 2024-04-03 110535](https://github.com/Vinishofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146931793/d4a6ac8f-1abe-4151-931a-cc4f4f9cf267)

### Prediction of LR:
![Screenshot 2024-04-03 110559](https://github.com/Vinishofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146931793/b47a5260-dfab-46e4-850c-046af86a291f)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
