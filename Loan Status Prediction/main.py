import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt




#load data
loan_dataset=pd.read_csv("dataset.csv")
#print(loan_dataset.head())


#statical measure
#loan_dataset.describe()

#missing values
#loan_dataset.isnull().sum()

#drop the missing values
loan_dataset=loan_dataset.dropna()

#label encoding
loan_dataset.replace({"Loan_Status":{"N":0,"Y":1}},inplace=True)

#dependent columns values
#loan_dataset['Dependents'].value_counts()

#replacing the 3+ to 4
loan_dataset=loan_dataset.replace(to_replace='3+',value=4)



#data visualization 
#sns.countplot(x="Education",hue="Loan_Status",data=loan_dataset)
#plt.show()


#convert categorical columns numerical values


loan_dataset.replace({"Married":{"No":0,"Yes":1},
                      "Gender":{"Male":1,"Female":0},
                      "Self_Employed":{"No":0,"Yes":1},
                      "Property_Area":{"Rural":0,"Semiurban":1,"Urban":2},
                      "Education":{"Graduate":1,"Not Graduate":0}},inplace=True)


#seperating the data
x=loan_dataset.drop(columns=["Loan_Status", "Loan_ID"])
y=loan_dataset["Loan_Status"]


#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,stratify=y, random_state=2)


#training the model
classifier=svm.SVC(kernel="linear")
classifier.fit(x_train, y_train)


#model Evaluation
#accuracy score

x_train_prediction=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
#print("accuracy score",training_data_accuracy)



#making a predictive system

input_data=(1,1,0,1,1,3000,0,66,360,1,2)
input_data_as_array=np.asarray(input_data)
input_data_reshaped=input_data_as_array.reshape(1,-1)
prediction=classifier.predict(input_data_reshaped)
if(prediction[0]==1):
    print("The person have approve the Loan")
else:
    print("The person have reject the Loan")




