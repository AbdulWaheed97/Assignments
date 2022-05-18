Name: ABDUL WAHEED
 Batch ID: 280921
####Diabetes#########

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r"F:\data science\Assignments\Decision Tree\Diabetes.csv")

data.isnull().sum()
data.dropna()
data.columns


data['Class variable'].unique()
data['Class variable'].value_counts()
colnames = list(data.columns)

predictors = colnames[:8]
target = colnames[8]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

###Company_Data###########
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

da = pd.read_csv(r"F:\data science\Assignments\Decision Tree\Company_Data.csv")

da.isnull().sum()
da.dropna()
da.columns
da = data.drop(["phone"], axis = 1)

# Converting into binary
lb = LabelEncoder()
da["ShelveLoc"] = lb.fit_transform(da["ShelveLoc"])
da["Urban"] = lb.fit_transform(da["Urban"])
da["default"]=lb.fit_transform(da["ShelveLoc"])

da['default'].unique()
da['default'].value_counts()
colnames = list(da.columns)

predictors = colnames[:10]
target = colnames[11]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(da, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy



#######Fraud_check#######

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r"F:\data science\Assignments\Decision Tree\Fraud_check.csv")

data.isnull().sum()
data.dropna()
data.columns

# Converting into binary
lb = LabelEncoder()
data["Undergrad"] = lb.fit_transform(data["Undergrad"])
data["Marital.Status"] = lb.fit_transform(data["Marital.Status"])



data['Urban'].unique()
data['Urban'].value_counts()
colnames = list(data.columns)

predictors = colnames[:5]
target = colnames[5]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy


############HR_DT##############
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r"F:\data science\Assignments\Decision Tree\HR_DT.csv")

data.isnull().sum()
data.dropna()
data.columns


data['Position of the employee'].unique()
data['Position of the employee'].value_counts()
colnames = list(data.columns)

predictors = colnames[1:]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

















