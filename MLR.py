






import pandas as pd
import numpy as np

cars  = pd.read_csv("C:/Users/Dell/Documents/50_Startups.csv")


cars.describe()

cars = cars.drop("State", axis = 1)


cars.columns = cars.columns.str.replace("R&D Spend", "A")
cars.columns = cars.columns.str.replace("Administration", "B")
cars.columns = cars.columns.str.replace("Marketing Spend", "C")
cars.columns = cars.columns.str.replace("Profit", "D")




#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#B
plt.bar(height = cars.B, x = np.arange(1, 51, 1))
plt.hist(cars.B) #histogram
plt.boxplot(cars.B) #boxplot

#A
plt.bar(height = cars.A, x = np.arange(1, 51, 1))
plt.hist(cars.A) #histogram
plt.boxplot(cars.A) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=cars['A'], y=cars['B'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(cars['B'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(cars.A, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cars.iloc[:, :])
                             
# Correlation matrix 
cars.corr()

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('B ~ A + C + D ', data = cars).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

cars_new = cars.drop(cars.index[[48]])

# Preparing model                  
ml_new = smf.ols('B ~ A+ C + D', data = cars_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_A = smf.ols('B ~ A + C + D', data = cars).fit().rsquared  
vif_A = 1/(1 - rsq_A) 

rsq_B= smf.ols('B ~ A + C + D', data = cars).fit().rsquared  
vif_B = 1/(1 - rsq_B)

rsq_C = smf.ols('B ~ A + C + D', data = cars).fit().rsquared  
vif_C = 1/(1 - rsq_C) 

rsq_D = smf.ols('B ~ A + C + D', data = cars).fit().rsquared  
vif_D = 1/(1 - rsq_D) 

# Storing vif values in a data frame
d1 = {'Variables':["A", "B", "C", "D"], 'VIF':[vif_A, vif_B, vif_C, vif_D]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('B ~ A + C + D', data = cars).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(cars)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = cars.A, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
cars_train, cars_test = train_test_split(cars, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('B ~ A + C + D', data = cars_train).fit()

# prediction on test data set 
test_pred = model_train.predict(cars_test)

# test residual values 
test_resid = test_pred - cars_test.B
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(cars_train)

# train residual values 
train_resid  = train_pred - cars_train.B
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse





#################################################################





import pandas as pd
import numpy as np

cars  = pd.read_csv("C:/Users/Dell/Documents/Avacado_Price.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

cars.describe()
cars.info()
cars = cars.drop('type', axis = 1)
cars = cars.drop("region", axis = 1)


cars.columns = cars.columns.str.replace("AveragePrice", "A")
cars.columns = cars.columns.str.replace("Total_Valume", "B")
cars.columns = cars.columns.str.replace("tot_ava1", "B")
cars.columns = cars.columns.str.replace("tot_ava2", "C")
cars.columns = cars.columns.str.replace("tot_ava3", "D")
cars.columns = cars.columns.str.replace("Total_Bags", "E")
cars.columns = cars.columns.str.replace("Small_Bags", "F")
cars.columns = cars.columns.str.replace("Large_Bags", "G")
cars.columns = cars.columns.str.replace("XLarge Bags", "H")
cars.columns = cars.columns.str.replace("year", "I")


#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#B
plt.bar(height = cars.B, x = np.arange(1, 1850, 1))
plt.hist(cars.B) #histogram
plt.boxplot(cars.B) #boxplot

#A
plt.bar(height = cars.A, x = np.arange(1, 18250, 1))
plt.hist(cars.A) #histogram
plt.boxplot(cars.A) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=cars['A'], y=cars['B'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(cars['B'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(cars.A, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cars.iloc[:, :])
                             
# Correlation matrix 
cars.corr()

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('B ~  + C + D + E + F + G + H + I', data = cars).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

cars_new = cars.drop(cars.index[[48]])

# Preparing model                  
ml_new = smf.ols('B ~  + C + D + E + F + G + H + I', data = cars_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables

rsq_A = smf.ols('B ~  + C + D + E + F + G + H + I', data = cars).fit().rsquared  
vif_A = 1/(1 - rsq_A) 

rsq_B= smf.ols('B ~  + C + D + E + F + G + H + I', data = cars).fit().rsquared  
vif_B = 1/(1 - rsq_B)

rsq_C = smf.ols('B ~  + C + D + E + F + G + H + I', data = cars).fit().rsquared  
vif_C = 1/(1 - rsq_C) 

rsq_D = smf.ols('B ~  + C + D + E + F + G + H + I', data = cars).fit().rsquared  
vif_D = 1/(1 - rsq_D) 


rsq_E = smf.ols('B ~  + C + D + E + F + G + H + I', data = cars).fit().rsquared  
vif_E = 1/(1 - rsq_E) 

rsq_F= smf.ols('B ~  + C + D + E + F + G + H + I', data = cars).fit().rsquared  
vif_F = 1/(1 - rsq_F)

rsq_G = smf.ols('B ~  + C + D + E + F + G + H + I', data = cars).fit().rsquared  
vif_G = 1/(1 - rsq_G) 


rsq_I = smf.ols('B ~  + C + D + E + F + G + H + I', data = cars).fit().rsquared  
vif_I = 1/(1 - rsq_I) 

# Storing vif values in a data frame
d1 = {'Variables':["A", "B", "C", "D" "E", "F", "G", "H", "I"], 'VIF':[vif_A, vif_B, vif_C, vif_D, vif_E, vif_F, vif_G, vif_I]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('B ~  + C + D + E + F + G + H + I'), data = cars).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(cars)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = cars.A, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
cars_train, cars_test = train_test_split(cars, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('B ~ A + C + D + E + F + G + H + I', data = cars_train).fit()

# prediction on test data set 
test_pred = model_train.predict(cars_test)

# test residual values 
test_resid = test_pred - cars_test.B
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(cars_train)

# train residual values 
train_resid  = train_pred - cars_train.B
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse






#############################################################################################


#############################################################







import numpy as np
import pandas as pd

cars = pd.read_csv("C:/Users/Dell/Documents/Company_Data.csv")
cars.info()


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


cars['US']= labelencoder.fit_transform(cars['US'])
cars['Urban'] = labelencoder.fit_transform(cars['Urban'])
cars['ShelveLoc'] = labelencoder.fit_transform(cars['ShelveLoc'])




cars.columns = cars.columns.str.replace("Sales", "A")
cars.columns = cars.columns.str.replace("CompPrice", "B")
cars.columns = cars.columns.str.replace("Income", "C")
cars.columns = cars.columns.str.replace("Advertising", "D")
cars.columns = cars.columns.str.replace("Population", "E")
cars.columns = cars.columns.str.replace("Price", "F")
cars.columns = cars.columns.str.replace("ShelveLoc", "G")
cars.columns = cars.columns.str.replace("Age", "H")
cars.columns = cars.columns.str.replace("Education", "I")
cars.columns = cars.columns.str.replace("Urban", "J")
cars.columns = cars.columns.str.replace("US", "K")

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#B
plt.bar(height = cars.B, x = np.arange(1, 401, 1))
plt.hist(cars.B) #histogram
plt.boxplot(cars.B) #boxplot

#A
plt.bar(height = cars.A, x = np.arange(1, 401, 1))
plt.hist(cars.A) #histogram
plt.boxplot(cars.A) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=cars['A'], y=cars['C'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(cars['B'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(cars.A, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cars.iloc[:, :])
                             
# Correlation matrix 
cars.corr()

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('B ~  + A + C + D + E + F + G + H + I + J', data = cars).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

cars_new = cars.drop(cars.index[[48]])

# Preparing model                  
ml_new = smf.ols('B ~ + A + C + D + E + F + G + H + I + J', data = cars_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables

rsq_A = smf.ols('B ~ + A + C + D + E + F + G + H + I', data = cars).fit().rsquared  
vif_A = 1/(1 - rsq_A) 

rsq_B= smf.ols('B ~  + A + C + D + E + F + G + H + I + J', data = cars).fit().rsquared  
vif_B = 1/(1 - rsq_B)

rsq_C = smf.ols('B ~  + C + A + D + E + F + G + H + I + J', data = cars).fit().rsquared  
vif_C = 1/(1 - rsq_C) 

rsq_D = smf.ols('B ~  + A + C + D + E + F + G + H + I + J', data = cars).fit().rsquared  
vif_D = 1/(1 - rsq_D) 


rsq_E = smf.ols('B ~  + A + C + D + E + F + G + H + I + J', data = cars).fit().rsquared  
vif_E = 1/(1 - rsq_E) 

rsq_F= smf.ols('B ~  + A + C + D + E + F + G + H + I + J', data = cars).fit().rsquared  
vif_F = 1/(1 - rsq_F)

rsq_G = smf.ols('B ~  + A + C + D + E + F + G + H + I + J', data = cars).fit().rsquared  
vif_G = 1/(1 - rsq_G) 


rsq_I = smf.ols('B ~  + A + C + D + E + F + G + H + I + J', data = cars).fit().rsquared  
vif_I = 1/(1 - rsq_I) 


rsq_J = smf.ols('B ~  + A + C + D + E + F + G + H + I + J', data = cars).fit().rsquared  
vif_J= 1/(1 - rsq_J) 

# Storing vif values in a data frame
# Storing vif values in a data frame
d1 = {'Variables':["A", "B", "C", "D" "E", "F", "G", "H", "I", "J"], 'VIF':[vif_A, vif_B, vif_C, vif_D, vif_E, vif_F, vif_G, VIF_H vif_I, VIF_J]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('B ~  +A + C + D + E + F + G + H + I + J'), data = cars).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(cars)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = cars.A, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
cars_train, cars_test = train_test_split(cars, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('B ~ A + C + D + E + F + G + H + I + J + K', data = cars_train).fit()

# prediction on test data set 
test_pred = model_train.predict(cars_test)

# test residual values 
test_resid = test_pred - cars_test.B
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(cars_train)

# train residual values 
train_resid  = train_pred - cars_train.B
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse



#################################################################################




import numpy as np
import pandas as pd

cars = pd.read_csv(r"C:/Users/Dell/Documents/ToyotaCorolla.csv",encoding='ISO-8859-1')



# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

cars.describe()

cars = cars.drop("Model", axis = 1)

cars.info()

cars.columns = cars.columns.str.replace("Price", "A")
cars.columns = cars.columns.str.replace("Age_08_04", "B")
cars.columns = cars.columns.str.replace("Mfg_Month", "C")
cars.columns = cars.columns.str.replace("Mfg_Year", "D")




#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#B
plt.bar(height = cars.B, x = np.arange(1, 1437, 1))
plt.hist(cars.B) #histogram
plt.boxplot(cars.B) #boxplot

#A
plt.bar(height = cars.A, x = np.arange(1, 1437, 1))
plt.hist(cars.A) #histogram
plt.boxplot(cars.A) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=cars['A'], y=cars['B'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(cars['B'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(cars.A, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cars.iloc[:, :])
                             
# Correlation matrix 
cars.corr()

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('B ~ A + C + D ', data = cars).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

cars_new = cars.drop(cars.index[[48]])

# Preparing model                  
ml_new = smf.ols('B ~ A+ C + D', data = cars_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_A = smf.ols('B ~ A + C + D', data = cars).fit().rsquared  
vif_A = 1/(1 - rsq_A) 

rsq_B= smf.ols('B ~ A + C + D', data = cars).fit().rsquared  
vif_B = 1/(1 - rsq_B)

rsq_C = smf.ols('B ~ A + C + D', data = cars).fit().rsquared  
vif_C = 1/(1 - rsq_C) 

rsq_D = smf.ols('B ~ A + C + D', data = cars).fit().rsquared  
vif_D = 1/(1 - rsq_D) 

# Storing vif values in a data frame
d1 = {'Variables':["A", "B", "C", "D"], 'VIF':[vif_A, vif_B, vif_C, vif_D]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('B ~ A + C + D', data = cars).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(cars)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = cars.A, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
cars_train, cars_test = train_test_split(cars, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('B ~ A + C + D', data = cars_train).fit()

# prediction on test data set 
test_pred = model_train.predict(cars_test)

# test residual values 
test_resid = test_pred - cars_test.B
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(cars_train)

# train residual values 
train_resid  = train_pred - cars_train.B
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse




