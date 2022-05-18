

### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mode = pd.read_csv("C:/Users/Dell/Documents/mdata.csv")
mode.head(10)




mode =  mode.drop("Unnamed: 0", axis = 1)

mode = mode.iloc[:, [4,0,1,2,3,5,6,7,8,9]]

mode.info()
#######################

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

mode['female']= labelencoder.fit_transform(mode['female'])
mode['schtyp'] = labelencoder.fit_transform(mode['schtyp'])
mode['prog']= labelencoder.fit_transform(mode['prog'])
mode['honors'] = labelencoder.fit_transform(mode['honors'])

### label encode y ###
mode.describe()
mode.ses.value_counts()



mode = mode.iloc[:,[4,0,1,2,3,4,5,6,7,8,9]]

# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "ses", y = "id", data = mode)
sns.boxplot(x = "ses", y = "female", data = mode)
sns.boxplot(x = "ses", y = "schtyp", data = mode)
sns.boxplot(x = "ses", y = "prog", data = mode)
sns.boxplot(x = "ses", y = "read", data = mode)
sns.boxplot(x = "ses", y = "write", data = mode)
sns.boxplot(x = "ses", y = "math", data = mode)
sns.boxplot(x = "ses", y = "science", data = mode)
sns.boxplot(x = "ses", y = "honours", data = mode)


# Scatter plot for each categorical choice of car

sns.stripplot(x = "ses", y = "id", jitter = True, data = mode)
sns.stripplot(x = "ses", y = "female", jitter = True, data = mode)
sns.stripplot(x = "ses", y = "schtyp", jitter = True, data = mode)
sns.stripplot(x = "ses", y = "prog", jitter = True, data = mode)
sns.stripplot(x = "ses", y = "read", jitter = True, data = mode)
sns.stripplot(x = "ses", y = "write", jitter = True, data = mode)
sns.stripplot(x = "ses", y = "math", jitter = True, data = mode)
sns.stripplot(x = "ses", y = "science", jitter = True, data = mode)
sns.stripplot(x = "ses", y = "honours", jitter = True, data = mode)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(mode) # Normal
sns.pairplot(mode, hue = "ses") # With showing the category of each car choice in the scatter plot

# Correlation values between each independent features
mode.corr()

train, test = train_test_split(mode, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict) 







### Multinomial Regression 
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mode = pd.read_csv("C:/Users/Dell/Documents/loan.csv")

mode = mode.iloc[:, :15]

mode = mode.drop(["id","member_id","int_rate"], axis = 1)
mode = mode.drop(["emp_title"], axis = 1)




from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
X = mode

X["term"]= labelencoder.fit_transform(X['term'])
X['grade'] = labelencoder.fit_transform(X['grade'])
X['sub_grade'] = labelencoder.fit_transform(X['sub_grade'])
X['emp_length']= labelencoder.fit_transform(X['emp_length'])
X['home_ownership'] = labelencoder.fit_transform(X['home_ownership'])
X = mode


X= X.iloc[:, [10,0,1,2,3,4,5,6,7,8,9]]

mode = X

X.info()
mode.verification_status.value_counts()

# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "verification_status", y = "loan_amnt", data = mode)
sns.boxplot(x = "verification_status", y = "funded-amnt", data = mode)
sns.boxplot(x = "verification_status", y = "funded_amnt_inv", data = mode)
sns.boxplot(x = "verification_status", y = "term", data = mode)
sns.boxplot(x = "verification_status", y = "installment", data = mode)    
sns.boxplot(x = "verification_status", y = "grade", data = mode)
sns.boxplot(x = "verification_status", y = "sub_grade", data = mode)
sns.boxplot(x = "verification_status", y = "emp_length", data = mode)


# Scatter plot for each categorical choice of car
sns.stripplot(x = "verification_status", y = "loan_amnt", jitter = True, data = mode)
sns.stripplot(x = "verification_status", y = "funded_amnt", jitter = True, data = mode)
sns.stripplot(x = "verification_status", y = "funded_amnt_inv", jitter = True, data = mode)
sns.stripplot(x = "verification_status", y = "term", jitter = True, data = mode)
sns.stripplot(x = "verification_status", y = "installment", jitter = True, data = mode)
sns.stripplot(x = "verification_status", y = "grade", jitter = True, data = mode)
sns.stripplot(x = "verification_status", y = "sub_grade", jitter = True, data = mode)
sns.stripplot(x = "verification_status", y = "emp_length", jitter = True, data = mode)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(mode) # Normal
sns.pairplot(mode, hue = "verification_status") # With showing the category of each car choice in the scatter plot

# Correlation values between each independent features
mode.corr()

train, test = train_test_split(mode, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict) 

###############################################################################



