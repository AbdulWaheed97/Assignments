#Name: ABDUL WAHEED
# Batch ID: _280921

#EastWestAirlines
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

da = pd.read_excel(r'C:\Users\Dell\Desktop\done assignments\EastWestAirlines.xlsx',1)
da.isna()
da.columns = da.columns.str.replace('ID#','SD')
da.columns= da.columns.str.replace('Award?','MD')
sns.boxplot(da.SD)
sns.boxplot(da.Balance)
#pip install feature_engine
from feature_engine.outliers import Winsorizer


sns.boxplot(da.SD)
sns.boxplot(da.Balance)
sns.boxplot(da.Qual_miles)
sns.boxplot(da.cc1_miles)
sns.boxplot(da.cc2_miles)
sns.boxplot(da.cc3_miles)
sns.boxplot(da.Bonus_miles)
sns.boxplot(da.Bonus_trans)
sns.boxplot(da.Flight_miles_12mo)
sns.boxplot(da.Flight_trans_12)
sns.boxplot(da.Days_since_enroll)
sns.boxplot(da.MD)
sns.boxplot(da.Bonus_trans)
winsor= Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Balance','Qual_miles','cc2_miles','cc3_miles','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12'])
df_fit= winsor.fit_transform(da[['Balance','Qual_miles','cc2_miles','cc3_miles','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12']])

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

df_norm = norm_func(da.iloc[:, 0:12])
df_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
ds['clust'] = cluster_labels


#         Autoinsurance

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
da = pd.read_csv(r'C:\Users\Dell\Desktop\done assignments\AutoInsurance.csv')
da.drop(da.columns[[0]],axis=1,inplace=True)
da.drop(da.columns[[5]],axis=1,inplace=True)
da.columns = da.columns.str.replace('Customer Lifetime Value','CN')
da.columns = da.columns.str.replace('Location Code','DN')
da.columns = da.columns.str.replace(' ','')
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
da['Coverage']= labelencoder.fit_transform(da['Coverage'])
da['State']= labelencoder.fit_transform(da['State'])
da['Response']= labelencoder.fit_transform(da['Response'])
da['Education']= labelencoder.fit_transform(da['Education'])
da['EmploymentStatus']= labelencoder.fit_transform(da['EmploymentStatus'])
da['Gender']= labelencoder.fit_transform(da['Gender'])
da['DN']= labelencoder.fit_transform(da['DN'])
da['MaritalStatus']= labelencoder.fit_transform(da['MaritalStatus'])
da['PolicyType']= labelencoder.fit_transform(da['PolicyType'])
da['Policy']= labelencoder.fit_transform(da['Policy'])
da['RenewOfferType']= labelencoder.fit_transform(da['RenewOfferType'])
da['SalesChannel']= labelencoder.fit_transform(da['SalesChannel'])
da['VehicleClass']= labelencoder.fit_transform(da['VehicleClass'])
da['VehicleSize']= labelencoder.fit_transform(da['VehicleSize'])

import seaborn as sns
sns.boxplot(da.State)
sns.boxplot(da.CN)
sns.boxplot(da.Response)
sns.boxplot(da.Coverage)
sns.boxplot(da.Education)
sns.boxplot(da.EmploymentStatus)
sns.boxplot(da.Gender)
sns.boxplot(da.Income)
sns.boxplot(da.DN)
sns.boxplot(da.MaritalStatus)
sns.boxplot(da.MonthlyPremiumAuto)
sns.boxplot(da.MonthsSinceLastClaim)
sns.boxplot(da.MonthsSincePolicyInception)
sns.boxplot(da.NumberofPolicies)
sns.boxplot(da.NumberofOpenComplaints)
sns.boxplot(da.PolicyType)
sns.boxplot(da.Policy)
sns.boxplot(da.RenewOfferType)
sns.boxplot(da.SalesChannel)
sns.boxplot(da.TotalClaimAmount)
sns.boxplot(da.VehicleClass)
sns.boxplot(da.VehicleSize)
from feature_engine.outliers import Winsorizer
winsor= Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['CN','Response','DN','NumberofOpenComplaints','PolicyType','TotalClaimAmount','VehicleSize'])
df_fit= winsor.fit_transform(da[['CN','Response','DN','NumberofOpenComplaints','PolicyType','TotalClaimAmount','VehicleSize']])


def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

df_norm= norm_func(da.iloc[:, 0:22])
df_norm.describe()
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
z = linkage(df_norm, method = "complete", metric = "euclidean")
z.shape()
plt.figure(figsize=(9134, 22));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
ds['clust'] = cluster_labels
dt = ds.iloc[:, [5,0,1,2,3,4]]
dt.head()


 #            Crime data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
ds=pd. read_csv(r'C:\Users\Dell\Desktop\done assignments)\crime_data.csv')
ds.columns= ds.columns.str.replace('Unnamed: 0','df')
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
ds['df']= labelencoder.fit_transform(ds['df'])

import seaborn as sns
sns.boxplot(ds.df)
sns.boxplot(ds.Murder)
sns.boxplot(ds.Assault)
sns.boxplot(ds.UrbanPop)
sns.boxplot(ds.Rape)
pip install feature_engine
from feature_engine.outliers import Winsorizer
winser= Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Rape'])
ds_fit =winser.fit_transform(ds[['Rape']])

def norm_func(i):
    x= (i-i.min())/(i-i.max()-(i-i.min()))
    return x
df_norm = norm_func(ds.iloc[:,0:49])
df_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
z = linkage(df_norm, method = "complete", metric = "euclidean")
plt.figure(figsize=(50,5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
ds['clust'] = cluster_labels
dt = ds.iloc[:, [5,0,1,2,3,4]]
dt.head()

 #            Telco_cluster
import pandas as pd 
import numpy as np 
import matplotlib.pylab as plt
import seaborn as sns
dh =pd.read_excel(r'C:\Users\Dell\Desktop\done assignments\Telco_customer_churn.xlsx')
dh.columns = dh.columns.str.replace(' ','')
dh.drop(dh.columns[[0]],axis=1,inplace=True)
dh.drop(dh.columns[[0]],axis=1,inplace=True)
dh.columns = dh.columns.str.replace('AvgMonthlyLongDistanceCharges','DS')
dh.columns = dh.columns.str.replace('AvgMonthlyGBDownload','MA')
from sklearn.preprocessing import LabelEncoder
labelencoder= LabelEncoder()
dh['Quarter']=labelencoder.fit_transform(dh['Quarter'])
dh['ReferredaFriend']=labelencoder.fit_transform(dh['ReferredaFriend'])
dh['Offer']=labelencoder.fit_transform(dh['Offer'])
dh['PhoneService']=labelencoder.fit_transform(dh['PhoneService'])
dh['MultipleLines']=labelencoder.fit_transform(dh['MultipleLines'])
dh['InternetService']=labelencoder.fit_transform(dh['InternetService'])
dh['InternetType']=labelencoder.fit_transform(dh['InternetType'])
dh['OnlineSecurity']=labelencoder.fit_transform(dh['OnlineSecurity'])
dh['OnlineBackup']=labelencoder.fit_transform(dh['OnlineBackup'])
dh['DeviceProtectionPlan']=labelencoder.fit_transform(dh['DeviceProtectionPlan'])
dh['PremiumTechSupport']=labelencoder.fit_transform(dh['PremiumTechSupport'])
dh['StreamingTV']=labelencoder.fit_transform(dh['StreamingTV'])
dh['StreamingMovies']=labelencoder.fit_transform(dh['StreamingMovies'])
dh['StreamingMusic']=labelencoder.fit_transform(dh['StreamingMusic'])
dh['UnlimitedData']=labelencoder.fit_transform(dh['UnlimitedData'])
dh['Contract']=labelencoder.fit_transform(dh['Contract'])
dh['PaperlessBilling']=labelencoder.fit_transform(dh['PaperlessBilling'])
dh['PaymentMethod']=labelencoder.fit_transform(dh['PaymentMethod'])


import seaborn as sns

sns.boxplot(dh.NumberofReferrals)
sns.boxplot(dh.PhoneService)
sns.boxplot(dh.MultipleLines)
sns.boxplot(dh.MA)
sns.boxplot(dh.TotalRefunds)
sns.boxplot(dh.TotalExtraDataCharges)
sns.boxplot(dh.TotalLongDistanceCharges)
sns.boxplot(dh.TotalRevenue)
pip install feature_engine
from feature_engine.outliers import Winsorizer
winser= Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['NumberofReferrals','PhoneService','MultipleLines','MA','TotalRefunds','TotalExtraDataCharges','TotalRevenue'])
ds_fit =winser.fit_transform(dh[['NumberofReferrals','PhoneService','MultipleLines','MA','TotalRefunds','TotalExtraDataCharges','TotalRevenue']])

def norm_func(i):
    x= (i-i.min())/(i-i.max()-(i-i.min()))
    return x
df_norm = norm_func(dh.iloc[:,1:28])
df_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
z = linkage(df_norm, method = "complete", metric = "euclidean")
plt.figure(figsize=(7043,28));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
dh['clust'] = cluster_labels

