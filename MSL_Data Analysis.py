# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 02:06:26 2018

@author: purve
"""

# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

#Importing the dataset after dropping extra columns
RSA_MSL = pd.read_csv('Data for Analysis.csv',encoding = "ISO-8859-1",low_memory=False)
#The attributes and instances count
RSA_MSL.shape


# Since we are using cognitive skills and leadership efficacy attributes only. 
# We create heatmaps to see if the attributes are correlated with one another.
# Incase, the attributes are correlated we will use one of them.
# 
# 
# 
# Definition of columns as citied in the code handbook provoded to us by RSA-
# #Cognitive Skills Pretest - 
# 1=Not At All Confident
# 2=Somewhat Confident
# 3=Confident
# 4=Very Confident
# 
# #PRE1a Handling the challenge of college-level work 
# #PRE1b Analyzing new ideas and concepts Cognitive Skills Pretest
# #PRE1c Applying something learned in class to the “real world”
# #PRE1d Enjoying the challenge of learning new material Cognitive Skills Pretest
# #PRE1e Appreciating new and different ideas or beliefs Cognitive Skills Pretest
# 
# #Leadership Efficacy Pretest
# #PRE2a Leading others Leadership Efficacy Pretest
# #PRE2b Organizing a group’s tasks to accomplish a goal Leadership Efficacy Pretest
# #PRE2c Taking initiative to improve something Leadership Efficacy Pretest
# #PRE2d Working with a team on a group project Leadership Efficacy Pretest

NumericAttributes = RSA_MSL[['PRE1A','PRE1B','PRE1C','PRE1D','PRE1E','PRE2A','PRE2B','PRE2C','PRE2D','SRLS4','SRLS3','SRLS52','SRLS33','SRLS29','SRLS5','SRLS63','SRLS30',
                             'SRLS40','SRLS28','SRLS22','SRLS42','SRLS13','SRLS23','SRLS48','SRLS24','SRLS16','SRLS34','SRLS47','SRLS51','SRLS53','SRLS41','SRLS59','SRLS60','SRLS54','SRLS62','SRLS27','SRLS10','SRLS69','SRLS9','SRLS66','SRLS1','SRLS32','SRLS71',
                             'REC1','REC2','REC3','REC4','REC5',
                             'OUT1A','OUT1B','OUT1C','OUT1D','OUT2A','OUT2B','OUT2C','OUT2D','PRECOG','PREEFF','PRESRLS','OMNIBUS','OUTCOG','OUTEFF']] 

CategoricalAttributes = RSA_MSL[['DEM7','ENV12','DEM3','DEM10A','ENV4A','ENV4B','ENV4C',
'ENV4D','ENV4E','ENV4F','ENV4G',
'ENV7A','ENV7F','ENV7G','ENV7B','ENV7C','ENV7P','ENV7Q','ENV7E','ENV7N',
'ENV7D','ENV7H','ENV7I','ENV7J','ENV7K','ENV7L','ENV7M','ENV7O','ENV7R','ENV7U','ENV7V','ENV7W','ENV7D1','ENV7D2','ENV7D3']]


# In[6]:


for column in NumericAttributes:
  #Replacing Null Values
    NumericAttributes[column].replace(['#NULL!'], ['-1'], inplace=True)
    NumericAttributes[column]=pd.to_numeric(NumericAttributes[column])

    


for column in CategoricalAttributes:
  #Replacing Null with NA
    CategoricalAttributes[column].replace(['#NULL!'], ['NA'], inplace=True)
    CategoricalAttributes[column].replace(r'\s+','NA',regex=True)
    


#Considering cognitive skills and leadership efficacy attributes to see the correlation between them

PreCog=NumericAttributes[['PRE1A','PRE1B','PRE1C','PRE1D','PRE1E' ]]

PreLeadershipEfficacy=NumericAttributes[['PRE2A','PRE2B','PRE2C','PRE2D']]

PostCog = NumericAttributes[['OUT1A','OUT1B','OUT1C','OUT1D']]

PostLeadershipEfficacy = NumericAttributes[['OUT2A','OUT2B','OUT2C','OUT2D']]

corrPreCog=PreCog.corr()

corrPreLeadershipEfficacy = PreLeadershipEfficacy.corr()

corrPostCog=PostCog.corr()

corrPostLeadershipEfficacy=PostLeadershipEfficacy.corr()


fig, (ax) = plt.subplots(1, 1, figsize=(10,6))

hm = sns.heatmap(corrPreCog, 
                 ax=ax,           # Axes in which to draw the plot, otherwise use the currently-active Axes.
                 cmap="coolwarm", # Color Map.
                 #square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
                 annot=True, 
                 fmt='.2f',       # String formatting code to use when adding annotations.
                 #annot_kws={"size": 14},
                 linewidths=.05)

fig, (ax) = plt.subplots(1, 1, figsize=(10,6))


hm = sns.heatmap(corrPreLeadershipEfficacy, 
                 ax=ax,           # Axes in which to draw the plot, otherwise use the currently-active Axes.
                 cmap="coolwarm", # Color Map.
                 #square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
                 annot=True, 
                 fmt='.2f',       # String formatting code to use when adding annotations.
                 #annot_kws={"size": 14},
                 linewidths=.05)


fig, (ax) = plt.subplots(1, 1, figsize=(10,6))


hm = sns.heatmap(corrPostCog, 
                 ax=ax,           # Axes in which to draw the plot, otherwise use the currently-active Axes.
                 cmap="coolwarm", # Color Map.
                 #square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
                 annot=True, 
                 fmt='.2f',       # String formatting code to use when adding annotations.
                 #annot_kws={"size": 14},
                 linewidths=.05)

fig, (ax) = plt.subplots(1, 1, figsize=(10,6))


hm = sns.heatmap(corrPostLeadershipEfficacy, 
                 ax=ax,           # Axes in which to draw the plot, otherwise use the currently-active Axes.
                 cmap="coolwarm", # Color Map.
                 #square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
                 annot=True, 
                 fmt='.2f',       # String formatting code to use when adding annotations.
                 #annot_kws={"size": 14},
                 linewidths=.05)


# Since, all the Precognitive determining attributes are highly correlated with each other we can use the mean of these attributes
# which is column 'PRECOG' in our survey data.
# 
# Similarly, all the leadership efficacy pretest variables, cognitieve skills post college variables and leadership efficacy post college variables are highly correlated so we'll use value 'PREEFF','OUTCOG' and 'POSTEFF' respectively from the data.
# 
# 


NumericAttributes.drop(['PRE1A','PRE1B','PRE1C','PRE1D','PRE1E','PRE2A','PRE2B','PRE2C','PRE2D','OUT1A','OUT1B','OUT1C','OUT1D',
              'OUT2A','OUT2B','OUT2C','OUT2D'], axis=1, inplace=True)


#Creating dataset for t-statistics, plotting and preliminary analysis
d_RSA_MSL = pd.concat([NumericAttributes,CategoricalAttributes], axis = 1)


# Description of Demgraphic Variables:
# 
# Variable Name : Dem3 -> 
# What is your current class level?
# 
# 1=Freshman/ First-year
# 2=Sophomore
# 3=Junior
# 4=Senior (4th year and beyond)
# 5=Graduate Student
# 6= Unclassified
# 
# Variable Name : Dem7 -> Gender
# What is your gender? 
# 
# 1=Genderqueer/ Gender NonConforming/Non-Binary
# 2=Man
# 3=Questioning/Unsure
# 4=Transgender
# 5=Woman
# 6= Preferred Response Not Listed
# 
# Variable Name : ENV12 ->
# Which of the following best describes where you
# are currently living while attending college?
# 
# 1= Off-campus with partner,
# spouse, and/ or children
# 2= Off-campus with
# parent/guardian or other relative
# 3=Other off-campus home,
# apartment, or room
# 4=College/university residence
# hall
# 5= Other on-campus student
# housing
# 6= Fraternity or sorority house
# 7=Other
# 
# 
# 
# PREEFF,OUTEFF, PRECOG, OUTCOG - Measurement Scale :
# 1=Not at all confident
# 2=Somewhat confident
# 3=Confident
# 4=Very confident

# In[16]:


#Comparing cognitive skills before and after college by Class Level

pd.pivot_table(d_RSA_MSL,index=["DEM3"],values=["PRECOG","OUTCOG"],aggfunc=np.mean)


#Comparing Leadership Efficacy before and after college by Class Level

pd.pivot_table(d_RSA_MSL,index=["DEM3"],values=["PREEFF","OUTEFF"],aggfunc=np.mean)

#Comparing cognitive skills before and after college by Gender

pd.pivot_table(d_RSA_MSL,index=["DEM7"],values=["PRECOG","OUTCOG"],aggfunc=np.mean)



#Comparing Leadership Efficacy before and after college by Gender

pd.pivot_table(d_RSA_MSL,index=["DEM7"],values=["PREEFF","OUTEFF"],aggfunc=np.mean)



#Comparing cognitive skills before and after college by Ethnicity

pd.pivot_table(d_RSA_MSL,index=["DEM7"],values=["PRECOG","OUTCOG"],aggfunc=np.mean)


#Comparing Leadership Efficacy before and after college by Ethnicity

pd.pivot_table(d_RSA_MSL,index=["DEM7"],values=["PREEFF","OUTEFF"],aggfunc=np.mean)


#t-statistics for overall data
PreCogT = stats.ttest_1samp(d_RSA_MSL['PRECOG'], 0)   
OutCogT = stats.ttest_1samp(d_RSA_MSL['OUTCOG'], 0)   
PreEffT = stats.ttest_1samp(d_RSA_MSL['PREEFF'], 0)   
OutEffT = stats.ttest_1samp(d_RSA_MSL['OUTEFF'], 0)   



print(PreCogT)
print(OutCogT)
print(PreEffT)
print(OutEffT)


from pandas.tools import plotting

plotting.scatter_matrix(d_RSA_MSL[['PRECOG', 'OUTCOG', 'PREEFF','OUTEFF','OMNIBUS']]) 



stats.ttest_ind(d_RSA_MSL['PRECOG'], d_RSA_MSL['OUTCOG']) 


dummies1=pd.get_dummies(d_RSA_MSL['DEM3'])
dummies1=dummies1.add_prefix("{}#".format('DEM3'))


dummies2=pd.get_dummies(d_RSA_MSL['DEM7'])
dummies2=dummies2.add_prefix("{}#".format('DEM7'))


dummies3=pd.get_dummies(d_RSA_MSL['ENV12'])
dummies3=dummies3.add_prefix("{}#".format('ENV12'))


data_RSA_MSL = pd.concat([dummies1,dummies2,dummies3,d_RSA_MSL], axis = 1)

data_RSA_MSL.drop(['DEM3','DEM7','DEM10A','ENV12'],axis=1,inplace=True)


#Linear regression with only DEM3 which is class level

from statsmodels.formula.api import ols


model_Dem3 = ols("OMNIBUS ~ data_RSA_MSL['DEM3#1']+data_RSA_MSL['DEM3#2']+data_RSA_MSL['DEM3#3']+data_RSA_MSL['DEM3#4']+ data_RSA_MSL['DEM3#5'] + data_RSA_MSL['DEM3#6'] + data_RSA_MSL['DEM3#NA']",data_RSA_MSL).fit()

print(model_Dem3.summary()) 

#Linear regression with only DEM7 which is Gender

model_Dem7 = ols("OMNIBUS ~ data_RSA_MSL['DEM7#1']+data_RSA_MSL['DEM7#2']+data_RSA_MSL['DEM7#3']+data_RSA_MSL['DEM7#4']+ data_RSA_MSL['DEM7#5'] + data_RSA_MSL['DEM7#6'] + data_RSA_MSL['DEM7#NA']",data_RSA_MSL).fit()
print(model_Dem7.summary()) 



#Linear Regression for Environment Variable - stating where the student resides suring their education

model_Env12 = ols("OMNIBUS ~ data_RSA_MSL['ENV12#1']+ data_RSA_MSL['ENV12#2']+ data_RSA_MSL['ENV12#3']+ data_RSA_MSL['ENV12#4']+ data_RSA_MSL['ENV12#5'] + data_RSA_MSL['ENV12#6'] + data_RSA_MSL['ENV12#7']+ data_RSA_MSL['ENV12#NA']",data_RSA_MSL).fit()

                  
print(model_Env12.summary()) 



#Linear regression using DEM3, DEM7, ENV12

model = ols("OMNIBUS ~ data_RSA_MSL['DEM3#1']+ data_RSA_MSL['DEM3#2']+data_RSA_MSL['DEM3#3']+data_RSA_MSL['DEM3#4']+ data_RSA_MSL['DEM3#5'] + data_RSA_MSL['DEM3#6'] + data_RSA_MSL['DEM3#NA'] + data_RSA_MSL['DEM7#1']+data_RSA_MSL['DEM7#2']+data_RSA_MSL['DEM7#3']+data_RSA_MSL['DEM7#4']+ data_RSA_MSL['DEM7#5'] + data_RSA_MSL['DEM7#6'] + data_RSA_MSL['DEM7#NA'] + data_RSA_MSL['ENV12#1']+ data_RSA_MSL['ENV12#2']+ data_RSA_MSL['ENV12#3']+ data_RSA_MSL['ENV12#4']+ data_RSA_MSL['ENV12#5'] + data_RSA_MSL['ENV12#6'] + data_RSA_MSL['ENV12#7']+ data_RSA_MSL['ENV12#NA']",data_RSA_MSL).fit()

print(model.summary()) 

d = data_RSA_MSL.copy()

#Clustering between Class Level and Pre college Cognitive Skills

x=CategoricalAttributes['DEM3']
y=NumericAttributes['PRECOG']
CategoricalAttributes['DEM3'].replace(['NA'], ['-1'], inplace=True)
CategoricalAttributes['DEM3']=pd.to_numeric(CategoricalAttributes['DEM3'])
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
plt.scatter(finalDf[finalDf.cluster==0]["DEM3"], finalDf[finalDf.cluster==0]["PRECOG"], s=50, c='red')
plt.scatter(finalDf[finalDf.cluster==1]["DEM3"], finalDf[finalDf.cluster==1]["PRECOG"], s=50, c='black')
plt.scatter(finalDf[finalDf.cluster==2]["DEM3"], finalDf[finalDf.cluster==2]["PRECOG"], s=50, c='blue')
plt.scatter(finalDf[finalDf.cluster==3]["DEM3"], finalDf[finalDf.cluster==3]["PRECOG"], s=50, c='pink')
plt.scatter(finalDf[finalDf.cluster==4]["DEM3"], finalDf[finalDf.cluster==4]["PRECOG"], s=50, c='pink')


#Clustering between Class Level and Post college Cognitive skills
x=CategoricalAttributes['DEM3']
y=NumericAttributes['OUTCOG']
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
plt.scatter(finalDf[finalDf.cluster==0]["DEM3"], finalDf[finalDf.cluster==0]["OUTCOG"], s=50, c='red')
plt.scatter(finalDf[finalDf.cluster==1]["DEM3"], finalDf[finalDf.cluster==1]["OUTCOG"], s=50, c='black')
plt.scatter(finalDf[finalDf.cluster==2]["DEM3"], finalDf[finalDf.cluster==2]["OUTCOG"], s=50, c='blue')
plt.scatter(finalDf[finalDf.cluster==3]["DEM3"], finalDf[finalDf.cluster==3]["OUTCOG"], s=50, c='pink')
plt.scatter(finalDf[finalDf.cluster==4]["DEM3"], finalDf[finalDf.cluster==4]["OUTCOG"], s=50, c='pink')

#Clustering Between Gender and Pre college Cognitive skills
x=CategoricalAttributes['DEM7']
y=NumericAttributes['PRECOG']
CategoricalAttributes['DEM7'].replace(['NA'], ['-1'], inplace=True)
CategoricalAttributes['DEM']=pd.to_numeric(CategoricalAttributes['DEM3'])
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
plt.scatter(finalDf[finalDf.cluster==0]["DEM7"], finalDf[finalDf.cluster==0]["PRECOG"], s=50, c='red')
plt.scatter(finalDf[finalDf.cluster==1]["DEM7"], finalDf[finalDf.cluster==1]["PRECOG"], s=50, c='black')
plt.scatter(finalDf[finalDf.cluster==2]["DEM7"], finalDf[finalDf.cluster==2]["PRECOG"], s=50, c='blue')
plt.scatter(finalDf[finalDf.cluster==3]["DEM7"], finalDf[finalDf.cluster==3]["PRECOG"], s=50, c='pink')
plt.scatter(finalDf[finalDf.cluster==4]["DEM7"], finalDf[finalDf.cluster==4]["PRECOG"], s=50, c='pink')

#Gender and Post college Cognitive skills
x=CategoricalAttributes['DEM7']
y=NumericAttributes['OUTCOG']
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
plt.scatter(finalDf[finalDf.cluster==0]["DEM7"], finalDf[finalDf.cluster==0]["OUTCOG"], s=50, c='red')
plt.scatter(finalDf[finalDf.cluster==1]["DEM7"], finalDf[finalDf.cluster==1]["OUTCOG"], s=50, c='black')
plt.scatter(finalDf[finalDf.cluster==2]["DEM7"], finalDf[finalDf.cluster==2]["OUTCOG"], s=50, c='blue')
plt.scatter(finalDf[finalDf.cluster==3]["DEM7"], finalDf[finalDf.cluster==3]["OUTCOG"], s=50, c='pink')
plt.scatter(finalDf[finalDf.cluster==4]["DEM7"], finalDf[finalDf.cluster==4]["OUTCOG"], s=50, c='pink')



#Student Accomodation and Pre college Cognitive skills

x=CategoricalAttributes['ENV12']
y=NumericAttributes['PRECOG']
CategoricalAttributes['ENV12'].replace(['NA'], ['-1'], inplace=True)
CategoricalAttributes['ENV12']=pd.to_numeric(CategoricalAttributes['ENV12'])
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
plt.scatter(finalDf[finalDf.cluster==0]["ENV12"], finalDf[finalDf.cluster==0]["PRECOG"], s=50, c='red')
plt.scatter(finalDf[finalDf.cluster==1]["ENV12"], finalDf[finalDf.cluster==1]["PRECOG"], s=50, c='black')
plt.scatter(finalDf[finalDf.cluster==2]["ENV12"], finalDf[finalDf.cluster==2]["PRECOG"], s=50, c='blue')
plt.scatter(finalDf[finalDf.cluster==3]["ENV12"], finalDf[finalDf.cluster==3]["PRECOG"], s=50, c='pink')
plt.scatter(finalDf[finalDf.cluster==4]["ENV12"], finalDf[finalDf.cluster==4]["PRECOG"], s=50, c='pink')


#Student Accomodation and Post college Cognitive skills


x=CategoricalAttributes['ENV12']
y=NumericAttributes['OUTCOG']
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
plt.scatter(finalDf[finalDf.cluster==0]["ENV12"], finalDf[finalDf.cluster==0]["OUTCOG"], s=50, c='red')
plt.scatter(finalDf[finalDf.cluster==1]["ENV12"], finalDf[finalDf.cluster==1]["OUTCOG"], s=50, c='black')
plt.scatter(finalDf[finalDf.cluster==2]["ENV12"], finalDf[finalDf.cluster==2]["OUTCOG"], s=50, c='blue')
plt.scatter(finalDf[finalDf.cluster==3]["ENV12"], finalDf[finalDf.cluster==3]["OUTCOG"], s=50, c='pink')
plt.scatter(finalDf[finalDf.cluster==4]["ENV12"], finalDf[finalDf.cluster==4]["OUTCOG"], s=50, c='pink')
#Student Accomodation and Post college Cognitive skills


x=CategoricalAttributes['ENV12']
y=NumericAttributes['OUTCOG']
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
plt.scatter(finalDf[finalDf.cluster==0]["ENV12"], finalDf[finalDf.cluster==0]["OUTCOG"], s=50, c='red')
plt.scatter(finalDf[finalDf.cluster==1]["ENV12"], finalDf[finalDf.cluster==1]["OUTCOG"], s=50, c='black')
plt.scatter(finalDf[finalDf.cluster==2]["ENV12"], finalDf[finalDf.cluster==2]["OUTCOG"], s=50, c='blue')
plt.scatter(finalDf[finalDf.cluster==3]["ENV12"], finalDf[finalDf.cluster==3]["OUTCOG"], s=50, c='pink')
plt.scatter(finalDf[finalDf.cluster==4]["ENV12"], finalDf[finalDf.cluster==4]["OUTCOG"], s=50, c='pink')


#Class Level and Pre college Leadership Efficacy
x=CategoricalAttributes['DEM3']
y=NumericAttributes['PREEFF']
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
plt.scatter(finalDf[finalDf.cluster==0]["DEM3"], finalDf[finalDf.cluster==0]["PREEFF"], s=50, c='red')
plt.scatter(finalDf[finalDf.cluster==1]["DEM3"], finalDf[finalDf.cluster==1]["PREEFF"], s=50, c='black')
plt.scatter(finalDf[finalDf.cluster==2]["DEM3"], finalDf[finalDf.cluster==2]["PREEFF"], s=50, c='blue')
plt.scatter(finalDf[finalDf.cluster==3]["DEM3"], finalDf[finalDf.cluster==3]["PREEFF"], s=50, c='pink')
plt.scatter(finalDf[finalDf.cluster==4]["DEM3"], finalDf[finalDf.cluster==4]["PREEFF"], s=50, c='pink')


#Class Level and Post college Leadership Efficacy

x=CategoricalAttributes['DEM3']
y=NumericAttributes['OUTEFF']
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
plt.scatter(finalDf[finalDf.cluster==0]["DEM3"], finalDf[finalDf.cluster==0]["OUTEFF"], s=50, c='red')
plt.scatter(finalDf[finalDf.cluster==1]["DEM3"], finalDf[finalDf.cluster==1]["OUTEFF"], s=50, c='black')
plt.scatter(finalDf[finalDf.cluster==2]["DEM3"], finalDf[finalDf.cluster==2]["OUTEFF"], s=50, c='blue')
plt.scatter(finalDf[finalDf.cluster==3]["DEM3"], finalDf[finalDf.cluster==3]["OUTEFF"], s=50, c='pink')
plt.scatter(finalDf[finalDf.cluster==4]["DEM3"], finalDf[finalDf.cluster==4]["OUTEFF"], s=50, c='pink')


#Gender and Pre college Leadership Efficacy
x=CategoricalAttributes['DEM7']
y=NumericAttributes['PREEFF']
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
plt.scatter(finalDf[finalDf.cluster==0]["DEM7"], finalDf[finalDf.cluster==0]["PREEFF"], s=50, c='red')
plt.scatter(finalDf[finalDf.cluster==1]["DEM7"], finalDf[finalDf.cluster==1]["PREEFF"], s=50, c='black')
plt.scatter(finalDf[finalDf.cluster==2]["DEM7"], finalDf[finalDf.cluster==2]["PREEFF"], s=50, c='blue')
plt.scatter(finalDf[finalDf.cluster==3]["DEM7"], finalDf[finalDf.cluster==3]["PREEFF"], s=50, c='pink')
plt.scatter(finalDf[finalDf.cluster==4]["DEM7"], finalDf[finalDf.cluster==4]["PREEFF"], s=50, c='pink')



#Gender and Post college LeadershipEfficacy

x=CategoricalAttributes['DEM7']
y=NumericAttributes['OUTEFF']
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
plt.scatter(finalDf[finalDf.cluster==0]["DEM7"], finalDf[finalDf.cluster==0]["OUTEFF"], s=50, c='red')
plt.scatter(finalDf[finalDf.cluster==1]["DEM7"], finalDf[finalDf.cluster==1]["OUTEFF"], s=50, c='black')
plt.scatter(finalDf[finalDf.cluster==2]["DEM7"], finalDf[finalDf.cluster==2]["OUTEFF"], s=50, c='blue')
plt.scatter(finalDf[finalDf.cluster==3]["DEM7"], finalDf[finalDf.cluster==3]["OUTEFF"], s=50, c='pink')
plt.scatter(finalDf[finalDf.cluster==4]["DEM7"], finalDf[finalDf.cluster==4]["OUTEFF"], s=50, c='pink')


#Student Accomodation and Pre college Leadership Efficacy

x=CategoricalAttributes['ENV12']
y=NumericAttributes['PREEFF']
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
plt.scatter(finalDf[finalDf.cluster==0]["ENV12"], finalDf[finalDf.cluster==0]["PREEFF"], s=50, c='red')
plt.scatter(finalDf[finalDf.cluster==1]["ENV12"], finalDf[finalDf.cluster==1]["PREEFF"], s=50, c='black')
plt.scatter(finalDf[finalDf.cluster==2]["ENV12"], finalDf[finalDf.cluster==2]["PREEFF"], s=50, c='blue')
plt.scatter(finalDf[finalDf.cluster==3]["ENV12"], finalDf[finalDf.cluster==3]["PREEFF"], s=50, c='pink')
plt.scatter(finalDf[finalDf.cluster==4]["ENV12"], finalDf[finalDf.cluster==4]["PREEFF"], s=50, c='pink')

#Student Accomodation and Post college Leadership Efficacy

x=CategoricalAttributes['ENV12']
y=NumericAttributes['OUTEFF']
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
plt.scatter(finalDf[finalDf.cluster==0]["ENV12"], finalDf[finalDf.cluster==0]["OUTEFF"], s=50, c='red')
plt.scatter(finalDf[finalDf.cluster==1]["ENV12"], finalDf[finalDf.cluster==1]["OUTEFF"], s=50, c='black')
plt.scatter(finalDf[finalDf.cluster==2]["ENV12"], finalDf[finalDf.cluster==2]["OUTEFF"], s=50, c='blue')
plt.scatter(finalDf[finalDf.cluster==3]["ENV12"], finalDf[finalDf.cluster==3]["OUTEFF"], s=50, c='pink')
plt.scatter(finalDf[finalDf.cluster==4]["ENV12"], finalDf[finalDf.cluster==4]["OUTEFF"], s=50, c='pink')


#Demographic and Pre College Cognitive Skills

x=CategoricalAttributes[['ENV12','DEM3','DEM7']]
y=NumericAttributes['PRECOG']
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
#print(finalDf)
print(finalDf.groupby(['cluster']).mean())


#Demographic and Post College Cognitive Skills


x=CategoricalAttributes[['ENV12','DEM3','DEM7']]
y=NumericAttributes['OUTCOG']
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
#print(finalDf)
print(finalDf.groupby(['cluster']).mean())



#Demographic and Pre College Leadership Efficacy

x=CategoricalAttributes[['ENV12','DEM3','DEM7']]
y=NumericAttributes['PREEFF']
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
#print(finalDf)
print(finalDf.groupby(['cluster']).mean())

#Demographic and Post College Leadership Efficacy


x=CategoricalAttributes[['ENV12','DEM3','DEM7']]
y=NumericAttributes['OUTEFF']
finalDf = pd.concat([x,y], axis = 1)
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5)
finalDf['cluster']=cluster.fit_predict(finalDf)
#print(finalDf)
print(finalDf.groupby(['cluster']).mean())


#Hierarchical Clustering

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

x=CategoricalAttributes['DEM7']
y=NumericAttributes['PREEFF']
finalDf = pd.concat([x,y], axis = 1)

#dendrogram = sch.dendrogram(sch.linkage(finalDf, method='ward'))
# create clusters
cluster = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
y_hc = cluster.fit_predict(finalDf)
print(y_hc)
plt.figure(figsize=(10, 7))  
plt.scatter(x, y, c=cluster.labels_, cmap='rainbow')  


#Hierarchical Clustering

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

x=CategoricalAttributes['DEM7']
y=NumericAttributes['OUTEFF']
finalDf = pd.concat([x,y], axis = 1)

#dendrogram = sch.dendrogram(sch.linkage(finalDf, method='ward'))
# create clusters
cluster = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
y_hc = cluster.fit_predict(finalDf)
print(y_hc)
plt.figure(figsize=(10, 7))  
plt.scatter(x, y, c=cluster.labels_, cmap='rainbow')  