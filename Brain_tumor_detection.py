import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df= pd.read_csv("C:/Deep_learning/Brain_tumor_data.csv")

# Perform EDA
print(df.shape)
print(df.info())
print(df.columns)
print(df.describe())

# drop column unnamed: 32 and id column
df= df.drop("Unnamed: 32" , axis= 1)
df= df.drop('id' , axis= 1)

column_list= list(df.columns)

features_mean= column_list[1 : 11]
features_se= column_list[11 : 21]
features_worst= column_list[21 : ]

# Check the distribution
print(df['diagnosis'].value_counts())
sns.countplot(df['diagnosis'] , label= "Count" ,)
plt.show()

# Map Malignant to 1 and Benign to 0
df['diagnosis']= df['diagnosis'].map({'M' : 1 , 'B' : 0})

# using the corr() method to find the correlation 
# between different fields
corr= df.corr()

# Perform heatmap
plt.figure(figsize= (14 , 14))
sns.heatmap(corr)
plt.show()

# Separate into feature and label set
X= df.drop('diagnosis' , axis= 1)
y= df['diagnosis']

# divide into test and train set
X_train , X_test , y_train , y_test= train_test_split(X ,
      y , test_size= 0.25)

ss= StandardScaler()
X_train= ss.fit_transform(X_train)
X_test= ss.transform(X_test)

# Use Various model measure accuracy 

# Linear Logistic Regression
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()
lr.fit(X_train , y_train)
y_pred= lr.predict(X_test)

print("Accuracy in Logistic Regression: {:.2f}".format(accuracy_score(y_test , y_pred)))

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dtc= DecisionTreeClassifier()
dtc.fit(X_train , y_train)
y_pred= dtc.predict(X_test)

print("Accuracy in Decision Tree Classifier: {:.2f}".format(accuracy_score(y_test , y_pred)))

# Random forest Classifier
from sklearn.ensemble import RandomForestClassifier

rfc= RandomForestClassifier()
rfc.fit(X_train , y_train)
y_pred= rfc.predict(X_test)

print("Accuracy in Random Forest Classifier: {:.2f}".format(accuracy_score(y_test , y_pred)))

# Support vector machine
from sklearn import svm

svc= svm.SVC()
svc.fit(X_train , y_train)
y_pred= svc.predict(X_test)

print("Accuracy in Support Vector Machine : {:.2f}".format(accuracy_score(y_test , y_pred)))
