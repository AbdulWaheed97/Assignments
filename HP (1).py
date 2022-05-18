NAME ABDUL WAHEED
BATCH ID 280921


import pandas as pd
import numpy as np
import scipy
from scipy import stats
prom = pd.read_csv('C:/Users/Dell/Documents/Cutlets.csv')
prom.columns = prom.columns.str.replace('Unit A', 'A')
prom.columns = prom.columns.str.replace('Unit B', 'B')
mean_value = prom.A.mean()
mean_value
prom.A = prom.A.fillna(mean_value)
prom.A.isna().sum()
mean_value = prom.B.mean()
mean_value
prom.B = prom.B.fillna(mean_value)
prom.B.isna().sum()
prom.columns = 'A', "B"

# Normality Test

stats.shapiro(prom.A)
stats.shapiro(prom.B)


# Variance test
scipy.stats.levene(prom.A, prom.B)


# 2 Sample T test
scipy.stats.ttest_ind(prom.A, prom.B)

###################################################



                                       
 import pandas as pd
import pandas as pd
import scipy
from scipy import stats

############ 2 sample T Test ##################

# Load the data
prom = pd.read_csv('C:/Users/Dell/Documents/lab_tat_updated.csv')
prom

prom.columns = "Laboratory_1", "Laboratory_2", "Labaratory_3", "Laboratory_4"

# Normality Test

stats.shapiro(prom.Laboratory_1)
stats.shapiro(prom.Laboratory_2)
stats.shapiro(prom.Laboratory_3)
stats.shapiro(prom.Laboratory_4)



# Variance test
scipy.stats.levene(prom.Laboratory_1, prom.Laboratory_2)


# 2 Sample T test
scipy.stats.ttest_ind(prom.Laboratory_1, prom.Laboratory_2)



#####################################################


import pandas as pd
import scipy
from scipy import stats

cof = pd.read_csv("C:/Users/Dell/Documents/BuyerRatio.csv")


cof.columns = cof.columns.str.replace("Observed Values", "ob")


cof.columns = "ob", "East", "West" "North", "South"
cof.info()



# Normality Test
stats.shapiro(cof.ob) # Shapiro Test
stats.shapiro(cof.East) # Shapiro Test
stats.shapiro(cof.West)
stats.shapiro(cof.North) # Shapiro Test
stats.shapiro(cof.South)
# Variance test
help(scipy.stats.levene)
# All 3 suppliers are being checked for variances
scipy.stats.levene( cof.North, cof.South, cof.West. cof.East)

# One - Way Anova
F, p = stats.f_oneway(cof.South, cof.West, cof.East, cof.North, cof.ob)

# p value
p  # P High Null Fly
# All the 3 suppliers have equal mean 


#####################################################################3333333

import pandas as pd
import scipy
from scipy import stats

Bahaman = pd.read_csv("C:/Users/Dell/Documents/CustomerOrderform.csv")
                   
Bahaman.info()


count = pd.crosstab(Bahaman["Malta"], Bahaman["India"])
count
Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square


###################################################

import pandas as pd
import scipy
from scipy import stats



Bahaman = pd.read_csv("C:/Users/Dell/Documents/Fantaloons.csv")
Bahaman
 

count = pd.crosstab(Bahaman["Weekdays"], Bahaman["Weekend"])
count
Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square
###################################################
