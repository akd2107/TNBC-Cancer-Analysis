

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle


# Read data into Python
s = pd.read_excel("C:/Users/damer/OneDrive/Desktop/MY PROJECT UPDATES/TNBC_survival (1).xlsx")


#Exploratory Data Analysis
#Measures of Central Tendency / First moment business decision
s.Age.mean()
s.surgerylevel.mean()
s.relapse_time.mean()
s.relapse.mean()
s.Outcome_time.mean()
s.event.mean()


s.Age.median()
s.surgerylevel.median()
s.relapse_time.median()
s.relapse.median()
s.Outcome_time.median()
s.event.median()


s.Age.mode()
s.surgerylevel.mode()
s.relapse_time.mode()
s.relapse.mode()
s.Outcome_time.mode()
s.event.mode()


# Measures of Dispersion / Second moment business decision
s.Age.var() 
s.surgerylevel.var()
s.relapse_time.var() 
s.relapse.var() 
s.Outcome_time.var() 
s.event.var() 


s.Age.std() 
s.surgerylevel.std()
s.relapse_time.std() 
s.relapse.std() 
s.Outcome_time.std() 
s.event.std() 


# Third moment business decision
s.Age.skew() 
s.surgerylevel.skew()
s.relapse_time.skew() 
s.relapse.skew() 
s.Outcome_time.skew() 
s.event.skew() 


# Fourth moment business decision
s.Age.kurt() 
s.surgerylevel.kurt()
s.relapse_time.kurt() 
s.relapse.kurt() 
s.Outcome_time.kurt() 
s.event.kurt()

 

# Data Visualization


# bar plot
plt.bar(height = s.Age, x = np.arange(1, 35, 1));plt.title('age')
plt.bar(height = s.surgerylevel, x = np.arange(1, 35, 1));plt.title('surgerylevel')
plt.bar(height = s.relapse_time, x = np.arange(1, 35, 1));plt.title('relapse_time')
plt.bar(height = s.relapse, x = np.arange(1, 35, 1));plt.title('relapse')
plt.bar(height = s.Outcome_time, x = np.arange(1, 35, 1));plt.title('Outcome_time')
plt.bar(height = s.event, x = np.arange(1, 35, 1));plt.title('event')


#histogram
plt.hist(s.Age);plt.title('Age') 
plt.hist(s.surgerylevel);plt.title('surgerylevel') 
plt.hist(s.relapse_time);plt.title('relapse_time') 
plt.hist(s.relapse);plt.title('relapse') 
plt.hist(s.Outcome_time);plt.title('Outcome_time') 
plt.hist(s.event);plt.title('event') 


# Normal Quantile-Quantile Plot
# Checking Whether data is normally distributed
stats.probplot(s.Age, dist="norm", plot=pylab);plt.title('Age')
stats.probplot(s.surgerylevel, dist="norm", plot=pylab);plt.title('surgerylevel')
stats.probplot(s.relapse_time, dist="norm", plot=pylab);plt.title('relapse_time')
stats.probplot(s.relapse, dist="norm", plot=pylab);plt.title('relapse')
stats.probplot(s.Outcome_time, dist="norm", plot=pylab);plt.title('Outcome_time')
stats.probplot(s.event, dist="norm", plot=pylab);plt.title('event')


#transformation to make workex variable normal
stats.probplot(np.log(s.Age),dist="norm",plot=pylab);plt.title('Age')
stats.probplot(np.log(s.surgerylevel),dist="norm",plot=pylab);plt.title('surgerylevel')
stats.probplot(np.log(s.relapse_time),dist="norm",plot=pylab);plt.title('relapse_time')
stats.probplot(np.log(s.relapse),dist="norm",plot=pylab);plt.title('relapse')
stats.probplot(np.log(s.Outcome_time),dist="norm",plot=pylab);plt.title('Outcome_time')
stats.probplot(np.log(s.event),dist="norm",plot=pylab);plt.title('event')

# Scatter plot between the variables along with histograms
sns.pairplot(s.iloc[:, :])

################################################
############## Data Preprocessing ##############

################################################
### Identify duplicates records in the data ####
duplicate = s.duplicated()
duplicate
sum(duplicate)

################################################
############## Outlier Treatment ###############
plt.boxplot(s.Age)
plt.title('Age')  # No outliers 


plt.boxplot(s.surgerylevel)
plt.title('surgerylevel')  # No outliers 

plt.boxplot(s.relapse_time)
plt.title('relapse_time')  # No outliers


plt.boxplot(s.relapse)
plt.title('relapse')  # No outliers 

plt.boxplot(s.Outcome_time)
plt.title('Outcome_time')  # No outliers 


plt.boxplot(s.event)
plt.title('event')  # No outliers 


##################################################
############  Label Encoder ###############
# creating instance of labelencoder

X = s.iloc[:, 0:13]

labelencoder = LabelEncoder()

X['HPE']= labelencoder.fit_transform(X['HPE'])
X['Stage'] = labelencoder.fit_transform(X['Stage'])
X['Tumor_Size'] = labelencoder.fit_transform(X['Tumor_Size'])
X['Surgery']= labelencoder.fit_transform(X['Surgery'])
X['Chemo_given_initially']= labelencoder.fit_transform(X['Chemo_given_initially'])
X['Treatment_given_on_relapse']= labelencoder.fit_transform(X['Treatment_given_on_relapse'])
X['Survival']= labelencoder.fit_transform(X['Survival'])

X.dtypes


###############################################################################
#################### Missing Values Imputation ################################
# check for count of NA'sin each column

X.isna().sum()
X = X.fillna(X.mean())


################################################
################# Type casting #################
# Now we will convert 'float64' into 'int64' type. 
X.relapse_time = X.relapse_time.astype('int64') 
X.Outcome_time = X.Outcome_time.astype('int64') 

X.dtypes


--# Normalized data frame (considering the numerical part of data)
--X = norm_func(X.iloc[:, 0:])
X.describe()

X = X.iloc[:, [0,1,2,3,4,5,6,8,9,10,11,12,7]]

X.describe()
X.columns

# Jointplot
import seaborn as sns
sns.jointplot(x=X['Age'], y=X['relapse'])
sns.jointplot(x=X['HPE'], y=X['relapse'])
sns.jointplot(x=X['Stage'], y=X['relapse'])
sns.jointplot(x=X['Tumor_Size'], y=X['relapse'])
sns.jointplot(x=X['Surgery'], y=X['relapse'])
sns.jointplot(x=X['surgerylevel'], y=X['relapse'])
sns.jointplot(x=X['relapse_time'], y=X['relapse'])
sns.jointplot(x=X['Chemo_given_initially'], y=X['relapse'])
sns.jointplot(x=X['Treatment_given_on_relapse'], y=X['relapse'])
sns.jointplot(x=X['Outcome_time'], y=X['relapse'])
sns.jointplot(x=X['Survival'], y=X['relapse'])
sns.jointplot(x=X['event'], y=X['relapse'])



# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(X['Age'])

plt.figure(1, figsize=(16, 10))
sns.countplot(X['HPE'])

plt.figure(1, figsize=(16, 10))
sns.countplot(X['Stage'])

plt.figure(1, figsize=(16, 10))
sns.countplot(X['Tumor_Size'])

plt.figure(1, figsize=(16, 10))
sns.countplot(X['Surgery'])

plt.figure(1, figsize=(16, 10))
sns.countplot(X['surgerylevel'])

plt.figure(1, figsize=(16, 10))
sns.countplot(X['relapse_time'])

plt.figure(1, figsize=(16, 10))
sns.countplot(X['Chemo_given_initially'])

plt.figure(1, figsize=(16, 10))
sns.countplot(X['Treatment_given_on_relapse'])

plt.figure(1, figsize=(16, 10))
sns.countplot(X['Outcome_time'])

plt.figure(1, figsize=(16, 10))
sns.countplot(X['Survival'])

plt.figure(1, figsize=(16, 10))
sns.countplot(X['event'])

# Scatter plot between the variables along with histograms
sns.pairplot(s.iloc[:, :])

# Correlation matrix 
X.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf 
         
ml1 = smf.ols('relapse ~ Age + HPE + Stage + Tumor_Size + Surgery + surgerylevel + relapse_time + Chemo_given_initially + Treatment_given_on_relapse + Outcome_time + Survival + event', data = X).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)


# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables

rsq_Age = smf.ols('Age ~ HPE + Stage + Tumor_Size + Surgery + surgerylevel + relapse_time + Chemo_given_initially + Treatment_given_on_relapse + Outcome_time + Survival + event', data = X).fit().rsquared  
vif_Age = 1/(1 - rsq_Age) 


rsq_HPE = smf.ols('HPE ~ Age + Stage + Tumor_Size + Surgery + surgerylevel + relapse_time + Chemo_given_initially + Treatment_given_on_relapse + Outcome_time + Survival + event', data = X).fit().rsquared  
vif_HPE = 1/(1 - rsq_HPE)


rsq_Stage = smf.ols('Stage ~ Age + HPE + Tumor_Size + Surgery + surgerylevel + relapse_time + Chemo_given_initially + Treatment_given_on_relapse + Outcome_time + Survival + event', data = X).fit().rsquared  
vif_Stage = 1/(1 - rsq_Stage) 


rsq_Tumor_Size = smf.ols('Tumor_Size ~ Age + HPE + Stage + Surgery + surgerylevel + relapse_time + Chemo_given_initially + Treatment_given_on_relapse + Outcome_time + Survival + event', data = X).fit().rsquared  
vif_Tumor_Size = 1/(1 - rsq_Tumor_Size) 


rsq_Surgery = smf.ols('Surgery ~ Age + HPE + Stage + Tumor_Size + surgerylevel + relapse_time + Chemo_given_initially + Treatment_given_on_relapse + Outcome_time + Survival + event', data = X).fit().rsquared  
vif_Surgery = 1/(1 - rsq_Surgery) 


rsq_surgerylevel = smf.ols('surgerylevel ~ Age + HPE + Stage + Tumor_Size + Surgery + relapse_time + Chemo_given_initially + Treatment_given_on_relapse + Outcome_time + Survival + event', data = X).fit().rsquared  
vif_surgerylevel = 1/(1 - rsq_surgerylevel)


rsq_relapse_time = smf.ols('relapse_time ~ Age + HPE + Stage + Tumor_Size + Surgery + surgerylevel + Chemo_given_initially + Treatment_given_on_relapse + Outcome_time + Survival + event', data = X).fit().rsquared  
vif_relapse_time = 1/(1 - rsq_relapse_time) 


rsq_Chemo_given_initially = smf.ols('Chemo_given_initially ~ Age + HPE + Stage + Tumor_Size + Surgery + surgerylevel + relapse_time + Treatment_given_on_relapse + Outcome_time + Survival + event', data = X).fit().rsquared  
vif_Chemo_given_initially = 1/(1 - rsq_Chemo_given_initially) 


rsq_Treatment_given_on_relapse = smf.ols('Treatment_given_on_relapse ~ Age + HPE + Stage + Tumor_Size + Surgery + surgerylevel + relapse_time + Chemo_given_initially + Outcome_time + Survival + event', data = X).fit().rsquared  
vif_Treatment_given_on_relapse = 1/(1 - rsq_Treatment_given_on_relapse) 


rsq_Outcome_time = smf.ols('Outcome_time ~ Age + HPE + Stage + Tumor_Size + Surgery + surgerylevel + relapse_time + Chemo_given_initially + Treatment_given_on_relapse + Survival + event', data = X).fit().rsquared  
vif_Outcome_time = 1/(1 - rsq_Outcome_time) 


rsq_Survival = smf.ols('Survival ~ Age + HPE + Stage + Tumor_Size + Surgery + surgerylevel + relapse_time + Chemo_given_initially + Treatment_given_on_relapse + Outcome_time + event', data = X).fit().rsquared  
vif_Survival = 1/(1 - rsq_Survival) 


rsq_event = smf.ols('event ~ Age + HPE + Stage + Tumor_Size + Surgery + surgerylevel + relapse_time + Chemo_given_initially + Treatment_given_on_relapse + Outcome_time + Survival', data = X).fit().rsquared  
vif_event = 1/(1 - rsq_event) 



# Storing vif values in a data frame
d1 = {'Variables':['Age', 'HPE', 'Stage', 'Tumor_Size', 'Surgery', 'surgerylevel', 'relapse_time', 'Chemo_given_initially', 'Treatment_given_on_relapse', 'Outcome_time', 'Survival', 'event' ], 'VIF':[vif_Age, vif_HPE, vif_Stage, vif_Tumor_Size, vif_Surgery, vif_surgerylevel, vif_relapse_time, vif_Chemo_given_initially, vif_Treatment_given_on_relapse, vif_Outcome_time, vif_Survival, vif_event]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('relapse ~ Age + HPE + Stage + Tumor_Size + Surgery + surgerylevel + relapse_time + Chemo_given_initially + Treatment_given_on_relapse', data = X).fit() 
final_ml.summary()

# Prediction
pred = final_ml.predict(X)

pred.astype('int64')



# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = X.relapse, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("relapse ~ Age + HPE + Stage + Tumor_Size + Surgery + surgerylevel + relapse_time + Chemo_given_initially + Treatment_given_on_relapse", data = X_train).fit()

# prediction on test data set 
test_pred = model_train.predict(X_test)

# test residual values 
test_resid = test_pred - X_test.relapse
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(X_train)

# train residual values 
train_resid  = train_pred - X_train.relapse
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse


pred = final_ml.predict(X)


# Saving model to disk
pickle.dump(final_ml, open('mlr.pkl','wb'))

# Loading model to compare the results
ml = pickle.load(open('mlr.pkl','rb'))

Z = X.iloc[1:2,:]

output=(ml.predict(Z))
print(round(output))


