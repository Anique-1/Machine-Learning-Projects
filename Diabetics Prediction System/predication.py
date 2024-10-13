#importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm



#Loading the data
diabetics_dataset=pd.read_csv("diabetes (1).csv")

#diabetics_dataset.head()
#no. of rows and columns
#diabetics_dataset.shape

#getting the statistical measure
#diabetics_dataset.describe()
#diabetics_dataset['Outcome'].value_counts()

#  0----> non diabetic
#  1----> diabetic



diabetics_dataset.groupby('Outcome').mean()


#seperating the data
x=diabetics_dataset.drop(columns='Outcome', axis=1)
y=diabetics_dataset['Outcome']

#print(x)
#print(y)


#data standardization
scaler=StandardScaler()
scaler.fit(x)
standardized_data=scaler.transform(x)

x=standardized_data
y=diabetics_dataset['Outcome']



#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)
#print(x.shape, x_train.shape, x_test.shape)



#training the model

classifer=svm.SVC(kernel='linear')
#training the svm
classifer.fit(x_train, y_train)


#model evalueation
#accuracy score
x_train_predication=classifer.predict(x_train)
training_accuracy=accuracy_score(x_train_predication, y_train)


#print("accuracy score", training_accuracy)

#making the prediction system
input_data=()
#changing the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)
#reshape the array
input_data_reshaped=input_data_as_numpy_array.reshape(1, -1)


#standardize the input data
std_data=scaler.transform(input_data_reshaped)
#print(std_data)
prediction=classifer.predict(std_data)
print(prediction)
if(prediction[0]==0):
    print("The Person is not Diabetic")
else:
    print("The person is Diabetic")




#saving the model
import pickle
filename='diabetic trained.csv'
pickle.dump(classifer, open(filename, 'wb')) 