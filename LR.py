###Affairs######

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


claimants = pd.read_csv("C:/Users/Dell/Documents/Affairs.csv")


claimants = claimants.drop("Unnamed: 0", axis = 1)
c1 = claimants
claimants.info()

claimants.columns = claimants.columns.str.replace("naffairs", "A")
claimants.columns = claimants.columns.str.replace("kids", "B")
claimants.columns = claimants.columns.str.replace("vryunhap", "C")
claimants.columns = claimants.columns.str.replace("unhap", "D")
claimants.columns = claimants.columns.str.replace("avgmarr", "E")
claimants.columns = claimants.columns.str.replace("hapavg", "F")
claimants.columns = claimants.columns.str.replace("vryhap", "G")
claimants.columns = claimants.columns.str.replace("antirel", "H")
claimants.columns = claimants.columns.str.replace("notrel", "I")
claimants.columns = claimants.columns.str.replace("yrsmarr6", "J")

c1 = claimants


########## Median Imputation for all the columns ############
c1.fillna(c1.median(), inplace=True)
c1.isna().sum()

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit("J ~ A + B + C + D + E + F + G + H + I" , data = c1).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(c1.iloc[ :, : ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(c1.J, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
c1["pred"] = np.zeros(601)
# taking threshold value and above the prob value will be treated as correct value 
c1.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(c1["pred"], c1["J"])
classification


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(c1, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit("J ~ A + B + C + D + E + F + G + H + I", data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(181)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['J'])
confusion_matrix

accuracy_test = (59 + 66)/(181) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["J"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["J"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, : ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(420)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['J'])
confusion_matrx

accuracy_train = (127 + 124)/(420)
print(accuracy_train)






#####################################################################

#Advertising######

import pandas as pd

claimants = pd.read_csv("C:/Users/Dell/Documents/advertising.csv")
claimants.info()
####################################
####################################################################################################
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

c1 = claimants

c1['Ad_Topic_Line']= labelencoder.fit_transform(c1['Ad_Topic_Line'])
c1['City'] = labelencoder.fit_transform(c1['City'])
c1['Country'] = labelencoder.fit_transform(c1['Country'])
c1['Timestamp'] = labelencoder.fit_transform(c1['Timestamp'])

claimants.columns = claimants.columns.str.replace("Daily_Time_ Spent _on_Site", "A")
claimants.columns = claimants.columns.str.replace("Age", "B")
claimants.columns = claimants.columns.str.replace("Area_Income", "C")
claimants.columns = claimants.columns.str.replace("Daily Internet Usage", "D")
claimants.columns = claimants.columns.str.replace("Ad_Topic_Line", "E")
claimants.columns = claimants.columns.str.replace("City", "F")
claimants.columns = claimants.columns.str.replace("Male", "G")
claimants.columns = claimants.columns.str.replace("Country", "H")
claimants.columns = claimants.columns.str.replace("Timestamp", "I")
claimants.columns = claimants.columns.str.replace("Clicked_on_Ad", "J")

c1 = claimants


########## Median Imputation for all the columns ############
c1.fillna(c1.median(), inplace=True)
c1.isna().sum()

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit("J ~ A + B + C + D + E + F + G + H + I" , data = c1).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(c1.iloc[ :, : ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(c1.J, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
c1["pred"] = np.zeros(1000)
# taking threshold value and above the prob value will be treated as correct value 
c1.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(c1["pred"], c1["J"])
classification


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(c1, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit("J ~ A + B + C + D + E + F + G + H + I", data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(300)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['J'])
confusion_matrix

accuracy_test = (146 + 145)/(402) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["J"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["J"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, : ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(700)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['J'])
confusion_matrx

accuracy_train = (351 + 330)/(938)
print(accuracy_train)






###############################################################################


#Election################




import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()




claimants = pd.read_csv("C:/Users/Dell/Documents/election_data.csv")


c1 = claimants
claimants.info()

claimants.columns = claimants.columns.str.replace("Election-id", "A")
claimants.columns = claimants.columns.str.replace("Result", "B")
claimants.columns = claimants.columns.str.replace("Year", "C")
claimants.columns = claimants.columns.str.replace("Amount Spent", "D")
claimants.columns = claimants.columns.str.replace("Popularity Rank", "E")


mean_value = c1.A.mean()
mean_value
c1.A= c1.A.fillna(mean_value)
c1.A.isna().sum()



mode_B = c1.B.mode()
mode_B
c1.B = c1.B.fillna((mode_B)[0])
c1.B.isna().sum()


mean_value = c1.C.mean()
mean_value
c1.C= c1.C.fillna(mean_value)
c1.C.isna().sum()


mean_value = c1.D.mean()
mean_value
c1.D= c1.D.fillna(mean_value)
c1.D.isna().sum()


mean_value = c1.E.mean()
mean_value
c1.E= c1.E.fillna(mean_value)
c1.E.isna().sum()


c1.fillna(c1.median(), inplace=True)
c1.isna().sum()

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit("B ~ A + C " , data = c1).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(c1.iloc[ :, : ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(c1.B, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
c1["pred"] = np.zeros(11)
# taking threshold value and above the prob value will be treated as correct value 
c1.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(c1["pred"], c1["B"])
classification


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(c1, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit("B ~ A  + C  ", data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(4)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['B'])
confusion_matrix

accuracy_test = (1+3)/(4) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["B"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["B"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, : ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(7)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['B'])
confusion_matrx

accuracy_train = (2 + 4)/(7)





##############################################################################################################

#bank#####

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


claimants = pd.read_csv("C:/Users/Dell/Documents/bank_data.csv")


c1 = claimants
claimants.info()

claimants.columns = claimants.columns.str.replace("age", "A")
claimants.columns = claimants.columns.str.replace("default", "B")
claimants.columns = claimants.columns.str.replace("balance", "C")
claimants.columns = claimants.columns.str.replace("housing", "D")
claimants.columns = claimants.columns.str.replace("loan", "E")
claimants.columns = claimants.columns.str.replace("duration", "F")
claimants.columns = claimants.columns.str.replace("campaign", "G")
claimants.columns = claimants.columns.str.replace("previous", "H")
claimants.columns = claimants.columns.str.replace("poutfailure", "I")
claimants.columns = claimants.columns.str.replace("y", "J")

c1 = claimants


########## Median Imputation for all the columns ############
c1.fillna(c1.median(), inplace=True)
c1.isna().sum()

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit("J ~ A + B + C + D + E + F + G + H + I" , data = c1).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(c1.iloc[ :, : ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(c1.J, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
c1["pred"] = np.zeros(45211)
# taking threshold value and above the prob value will be treated as correct value 
c1.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(c1["pred"], c1["J"])
classification


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(c1, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit("J ~ A + B + C + D + E + F + G + H + I", data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(13564)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['J'])
confusion_matrix

accuracy_test = (9409 + 1268)/(13564) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["J"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["J"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, : ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(31647)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['J'])
confusion_matrx

accuracy_train = (22042 + 2936)/(31647)
print(accuracy_train)





