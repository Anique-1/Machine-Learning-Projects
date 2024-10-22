#importing the libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




#Loading the data
sonar_data=pd.read_csv("Copy of sonar data.csv", header=None)

#sonar_data.head()
#no. of rows and columns
#sonar_data.shape

#how many rocks or mine
#sonar_data[60].value_counts()

# M-----> Mine
# R-----> Rock


sonar_data.groupby(60).mean()


#seperating the data
x=sonar_data.drop(columns=60, axis=1)
y=sonar_data[60]

#print(x)
#print(y)


#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.1, stratify=y, random_state=1)
#print(x.shape, x_train.shape, x_test.shape)



#training the model

model=LogisticRegression()
#training the svm
model.fit(x_train, y_train)


#model evalueation
#accuracy score
x_train_predication=model.predict(x_train)
training_accuracy=accuracy_score(x_train_predication, y_train)


#print("accuracy score", training_accuracy)

#making the prediction system
input_data=(0.0298,0.0615,0.065,0.0921,0.1615,0.2294,0.2176,0.2033,0.1459,0.0852,0.2476,0.3645,0.2777,0.2826,0.3237,0.4335,0.5638,0.4555,0.4348,0.6433,0.3932,0.1989,0.354,0.9165,0.9371,0.462,0.2771,0.6613,0.8028,0.42,0.5192,0.6962,0.5792,0.8889,0.7863,0.7133,0.7615,0.4401,0.3009,0.3163,0.2809,0.2898,0.0526,0.1867,0.1553,0.1633,0.1252,0.0748,0.0452,0.0064,0.0154,0.0031,0.0153,0.0071,0.0212,0.0076,0.0152,0.0049,0.02,0.0073)
#changing the input data to numpy array
input_data_as_numpy_array=np.array(input_data)
#reshape the array
input_data_reshaped=input_data_as_numpy_array.reshape(1, -1)

prediction=model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]=='R'):
    print("The Object is a Rock")
else:
    print("The Object is a Mine")




#saving the model
import pickle
filename='Rock vs Mine trained.csv'
pickle.dump(model, open(filename, 'wb')) 





