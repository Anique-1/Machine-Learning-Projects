import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



#loading the data
heart_dataset=pd.read_csv("heart_disease_data.csv")
print(heart_dataset.head())

#check missing value
#heart_dataset.isnull().sum() 

#statical measure
#heart_dataset.describe()

heart_dataset['target'].value_counts()
# 0------> Defective heart
# 1------> Healthy Heart


#splitting
x=heart_dataset.drop(columns="target",axis=1)
y=heart_dataset["target"]


#tarin test 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


#training the model
model=LogisticRegression()
model.fit(x_train,y_train)

#model evaluation
train_data_prediction=model.predict(x_train)
training_accuracy_score=accuracy_score(train_data_prediction,y_train)
#print(training_accuracy_score)


#making the pedictive system
input_data=(54,1,0,140,239,0,1,160,0,1.2,2,0,2)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)

if(prediction[0]==0):
    print("The person does not have heart disease")
else:
    print("The person has Heart Disease")
