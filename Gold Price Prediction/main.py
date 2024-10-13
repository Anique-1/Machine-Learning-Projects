import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics



#loading the data
gold_dataset=pd.read_csv("gld_price_data.csv")
print(gold_dataset.head())

#check missing value
#gold_dataset.isnull().sum() 

#statical measure
#gold_dataset.describe()

#correlation
#correlation=gold_dataset.corr()
#print(correlation)
#heat map
#plt.figure(figsize=(8,8))
#sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={"size":8},cmap="green")
#plt.show()

#checking distrubition
#sns.displot(gold_dataset['GLD'],color='green')
#plt.show()


#splitting
x=gold_dataset.drop(["Date","GLD"],axis=1)
y=gold_dataset["GLD"]


#tarin test 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


#training the model
regressor=RandomForestRegressor(n_estimators=100)
regressor.fit(x_train,y_train)

#model evaluation
test_data_prediction=regressor.predict(x_train)


#r_error
#error_score=metrics.r2_score(y_test,test_data_prediction)
#y_test=list(y_test)

