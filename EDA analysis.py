# %%
"""
# Importing Libraries

"""

# %%
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import os

# %%
"""
# Reading CSV File
"""

# %%
train = pd.read_csv('D:\\Github\\Titanic-Machine-Learning-from-Disaster\\dataset\\train.csv')
data = pd.read_csv('D:\\Github\\Titanic-Machine-Learning-from-Disaster\\dataset\\train.csv')
test = pd.read_csv('D:\\Github\\Titanic-Machine-Learning-from-Disaster\\dataset\\test.csv')

# %%
"""
# Dataset Print
"""

# %%
train.head()

# %%
"""
Variable Description <br><br>

1. Survived : 	Survival	    0 = No, 1 = Yes <br>
2. Pclass	:   Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd <br>
3. Sex	    :	Gender    <br>
4. Age	    :   Age in years	<br>
5. SibSp	:   # of siblings / spouses aboard the Titanic	<br>
6. Parch	:   # of parents / children aboard the Titanic	<br>
7. Ticket	:   Ticket number	<br>
8. Fare	    :   Passenger's ticket fare	 <br>
9. Cabin	:   Cabin number	<br>
10. Embarked:	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton <br>
"""

# %%
test.head()

# %%
"""
# Print the number of datapoints and features.
"""

# %%
train.shape

# %%
test.shape

# %%
"""
# Deleteing unecessary features
"""

# %%
del train['PassengerId']
del train['Name']
del train['Ticket']
del train['Cabin']



del test['PassengerId']
del test['Name']
del test['Ticket']
del test['Cabin']

# %%
"""
Feautures like PassengerId , Name , Ticket and cabin have no great influence on determining the survival of a person.
"""

# %%
test.shape

# %%
"""
# Checking unique values/variability of every columns altogether

"""

# %%
train.nunique()

# %%
"""
# Checking type of variable in the data set

"""

# %%
train.info()

# %%
"""
We can see that Age and Embarked  features have missing value and there is Sex , Survived , Embarked have categorical value.
"""

# %%
"""
# Datapoint per class
"""

# %%
train["Survived"] = train["Survived"].apply(lambda x: "Survived" if x == 1 else "Died")
train["Survived"].value_counts()

# %%
"""
It can be consiedered as a balanced dataset.
"""

# %%
"""
# Check for missing values
"""

# %%
train.isnull().sum()

# %%
test.isnull().sum()

# %%
"""
# Encoding categorical data
"""

# %%
"""
For encoding data: <br>
1) For Gender: <br>
    We replace male as 0 and female as 1 as it is binary classifier <br>
2) For Embarked: <br>
    We have three option (Q,C,S) so we create 3 columns each representing the embarked status in binary.<br>
    But in the set we dont include all 3 coluumns but only 2 to prevent dummy variable trap.
    
"""

# %%
#For train set
train["Sex"] = train["Sex"].apply(lambda x: 0 if x == "male" else 1)

train["Survived"] = train["Survived"].apply(lambda x: 1 if x == "Survived" else 0)

S = np.array(train["Embarked"].apply(lambda x: 1 if x == 'S' else 0))
C = np.array(train["Embarked"].apply(lambda x: 1 if x == 'C' else 0))
Q = np.array(train["Embarked"].apply(lambda x: 1 if x == 'Q' else 0))
S = S.reshape(train.shape[0],1)
C = C.reshape(train.shape[0],1)
Q = Q.reshape(train.shape[0],1)


train.insert(2,"Embarked(S)",S,True)
train.insert(2,"Embarked(C)",C,True)
del train["Embarked"]

train.head()


# %%
# For test set

test["Sex"] = test["Sex"].apply(lambda x: 0 if x == "male" else 1)

S = np.array(test["Embarked"].apply(lambda x: 1 if x == 'S' else 0))
C = np.array(test["Embarked"].apply(lambda x: 1 if x == 'C' else 0))
Q = np.array(test["Embarked"].apply(lambda x: 1 if x == 'Q' else 0))
S = S.reshape(test.shape[0],1)
C = C.reshape(test.shape[0],1)
Q = Q.reshape(test.shape[0],1)


test.insert(2,"Embarked(S)",S,True)
test.insert(2,"Embarked(C)",C,True)

del test["Embarked"]
test.head()

# %%
"""
# Filling missing data
"""

# %%
"""
For filling the missing value of the we have a standard method to use the mean or median of the data. <br>
But as we have missing values both in test and training set, this method will not solve the problem as this will only solve the problem for the training data and not for the testing data as we have to predict their survival status. <br> <br>

Another approach: <br>
Using the rest of the features we can train a simple regressor model which predicts the age of the person using rest of features ie. PClass, Sex,Fare,SibSp,Parch.<br><br>
Advantages of this approach: <br>
This model is not affected by the training set as it learns from the training set.

"""

# %%
x1 = train.loc[train.Age.notnull()]
test1 = train.loc[train.Age.isnull()]

y1  = x1.iloc[:,5]
x1 = x1.iloc[: , [1,2,3,4,6,7,8]]
test1 = test1.iloc[: , [1,2,3,4,6,7,8]]
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x1,y1)
y1_pred = regressor.predict(test1)
y1_pred[y1_pred <= 0] = np.min(y1)

age = train["Age"].copy()
age[np.isnan(age)] = y1_pred
train["Age"] = age
train["Age"] = train["Age"].astype(float)



# %%
train.isnull().sum()

# %%
test1 = test.loc[test.Age.isnull()]
test1 = test1.iloc[: , [0,1,2,3,5,6,7]]

y1_pred = regressor.predict(test1)
y1_pred[y1_pred <= 0] = np.min(y1)

age = test["Age"].copy()
age[np.isnan(age)] = y1_pred
test["Age"] = age
test["Age"] = test["Age"].astype(float)


# %%
test.isnull().sum()

# %%
test = test.fillna(train.Fare.mean())

# %%
"""
As we have only one missing value of Fare in test set we can use mean fare of train set to fill the value
"""

# %%
test.isnull().sum()

# %%
train.isnull().sum()

# %%
"""
No missing value in both Training and Test set.
"""

# %%
"""
# Dataset description
"""

# %%
train.describe()

# %%
sns.countplot(train['Survived'])

# %%
"""
# Checking Survival Status by a feature value
"""

# %%
"""
Inorder to understand who will have a better probability of survival, we can visualize the patients who survived based on age, passenger class and etc.
"""

# %%
train[['Embarked(C)','Survived']].groupby(['Embarked(C)']).mean().sort_values(by='Embarked(C)',ascending=True)

# %%
train[['Embarked(S)','Survived']].groupby(['Embarked(S)']).mean().sort_values(by='Embarked(S)',ascending=True)

# %%
train[['Pclass','Survived']].groupby(['Pclass']).mean().sort_values(by='Pclass',ascending=True)

# %%
"""
# Mean, Median, Percentile, IQR & MAD
"""

# %%
"""
## Mean and Median
"""

# %%
Survived = train.loc[train.Survived == 1]
Dead = train.loc[train.Survived == 0]

for i in ['Age','Fare']:
    print("\nMedian of",i,)
    print("Value is= ",np.median(Survived[i]))
    print("\nMean of",i,)
    print("Value is= ",np.mean(Survived[i]))

for i in ['Age','Fare']:
    print("\nMedian of",i,)
    print("Value is= ",np.median(Dead[i]))
    print("\nMean of",i,)
    print("Value is= ",np.mean(Dead[i]))

# %%
"""
As mean and median values are similiar no skew data is found in dataset
"""

# %%
"""
## Standard Deviation and MAD
"""

# %%
from statsmodels import robust
for i in ['Age','Fare']:
    print("\nStandard Deviation of",i,)
    print("Value is= ",np.std(Survived[i]))
    print("\nMean Absolute Deviation of",i)
    print("Value is= ",robust.mad(Survived[i]))
    
for i in ['Age','Fare']:
    print("\nStandard Deviation of",i,)
    print("Value is= ",np.std(Dead[i]))
    print("\nMean Absolute Deviation of",i)
    print("Value is= ",robust.mad(Dead[i])) 

# %%
"""
## Percentile and IQR
"""

# %%
for i in ['Age','Fare']:
    print("\nQuantiles of:",i)
    print(np.percentile(Survived[i],np.arange(0, 100, 25)))
    print("IQR of:",i)
    print(np.percentile(Survived[i],75)-np.percentile(Survived[i],25))
    

# %%
for i in ['Age','Fare']:
    print("\nQuantiles of:",i)
    print(np.percentile(Dead[i],np.arange(0, 100, 25)))
    print("IQR of:",i)
    print(np.percentile(Dead[i],75)-np.percentile(Dead[i],25))
    

# %%
"""
# Univariate Analysis 
"""

# %%
"""
## Histogram
"""

# %%
Pclass_Grid = sns.FacetGrid(train ,col = 'Survived')
Pclass_Grid.map(plt.hist,"Age",bins = 10)

Pclass_Grid = sns.FacetGrid(train,hue = 'Survived')
Pclass_Grid.map(sns.distplot,"Age" , rug = False , kde = True , hist = True)

# %%
"""
Observing the histogram we can see that large majority of people who could not survive belonged to age group 20 - 30

"""

# %%
Pclass_Grid = sns.FacetGrid(data ,col = 'Survived')
Pclass_Grid.map(plt.hist,"Embarked")

# %%
"""
Among those who did not survive, majority were of C. <br>
Similiarly among those who survived, mojority were of S
"""

# %%
Pclass_Grid = sns.FacetGrid(train ,col = 'Survived')
Pclass_Grid.map(plt.hist,"Pclass")

Pclass_Grid = sns.FacetGrid(train,hue = 'Survived')
Pclass_Grid.map(sns.distplot,"Pclass" , rug = False , kde = True , hist = True)

# %%
Pclass_Grid = sns.FacetGrid(train ,col = 'Survived')
Pclass_Grid.map(plt.hist,"Fare")

Pclass_Grid = sns.FacetGrid(train,hue = 'Survived')
Pclass_Grid.map(sns.distplot,"Fare" , rug = False , kde = True , hist = True)

# %%
Pclass_Grid = sns.FacetGrid(train ,col = 'Survived')
Pclass_Grid.map(plt.hist,"SibSp")

Pclass_Grid = sns.FacetGrid(train,hue = 'Survived')
Pclass_Grid.map(sns.distplot,"SibSp" , rug = False , kde = True , hist = True)

# %%
"""
## PDF and CDF 
"""

# %%
for i in ['Age','Fare','Parch','SibSp']:
    count, bin_edges= np.histogram(Survived[i])
    count2,bin_edges2=np.histogram(Dead[i])
    pdf_surv=count/sum(count)
    cdf_surv=np.cumsum(pdf_surv)
    pdf_dead=count2/sum(count2)
    cdf_dead=np.cumsum(pdf_dead)
    
    plt.title("PDF & CDF")
    plt.xlabel(i)
    plt.ylabel("Probability Density")
    plt.plot(bin_edges[1:],pdf_surv,label="PDF of survived person")
    plt.plot(bin_edges[1:],cdf_surv,label="CDF of survived person")
    plt.plot(bin_edges[1:],pdf_dead,label="PDF of dead person")
    plt.plot(bin_edges[1:],cdf_dead,label="CDF of dead person")

    plt.legend(loc="best")
    plt.show()
    

# %%
"""
## Box Plot
"""

# %%
sns.set_style("whitegrid")
sns.boxplot(x = train['Survived'] , y = train['Age'] , hue = train['Survived'] )

# %%
sns.set_style("whitegrid")
sns.boxplot(x = train['Survived'] , y = train['Fare'] , hue = train['Survived'] )

# %%
sns.set_style("whitegrid")
sns.boxplot(x = train['Survived'] , y = train['SibSp'] , hue = train['Survived'] )

# %%
"""
## Bar Plot
"""

# %%
for i in ['Embarked','Sex','Pclass']:
     sns.barplot(y="Survived",x=i,data=data)
     plt.show()

# %%
"""
People with PClass 1 , Embarked 'C' and are female have the highest chances of survival
"""

# %%
"""
## Violin Plot
"""

# %%
sns.set_style("whitegrid")
sns.violinplot(x = train['Survived'] , y = train['Age'] , hue = train['Survived'] )

# %%
"""
Very few people having age greater than 45 have survived
"""

# %%
sns.set_style("whitegrid")
sns.violinplot(x = train['Survived'] , y = train['Fare'] , hue = train['Survived'] )

# %%
sns.set_style("whitegrid")
sns.violinplot(x = train['Survived'] , y = train['Parch'] , hue = train['Survived'] )

# %%


# %%
"""
# BiVariate Analysis
"""

# %%
"""
## Pair Plot
"""

# %%
sns.pairplot(data = train ,vars = ['Age','Fare'] ,hue = 'Survived')

# %%
"""
The data are highly mixed up, none of the variable-pairs can help us find linearly separable clusters hence we can't find "lines" and "if-else" conditions to build a simple model to classify the survive status of the passenger.
"""

# %%
"""
## HeatMap 
"""

# %%
sns.heatmap(train.corr(),annot = True)

# %%
sns.heatmap(test.corr(),annot = True)

# %%
"""
Used to find any correlation betweeen data. No special correlation found.
"""

# %%
"""
## Contour Plots
"""

# %%
sns.jointplot(x = 'Age' , y = 'Fare' , data = Survived , kind = 'kde')
sns.jointplot(x = 'Age' , y = 'Fare' , data = Dead , kind = 'kde')


# %%
sns.jointplot(x = 'SibSp' , y = 'Parch' , data = Survived , kind = 'kde')

# %%
"""
1) Among survivors maximum people are those who have value of both Parch and SibSp as 0. <br> 
2) Among the non-survivors most of them were of age 25-35 and had farebetween 15 and 30.
"""

# %%
train.to_csv('D:\\Github\\Titanic-Machine-Learning-from-Disaster\\dataset\\updated.csv',columns = ['Survived', 'Pclass', 'Embarked(C)', 'Embarked(S)', 'Sex', 'Age',
       'SibSp', 'Parch'])


