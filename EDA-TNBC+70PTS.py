
#####importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

######uploading dataset 
data=pd.read_excel("C:/Users/damer/OneDrive/Desktop/MY PROJECT UPDATES/TNBC.xlsx")
####understanding the data
data.dtypes
data.head()
data.tail()
data.shape
data.describe
data.columns 
#####cleaning the data
data.isnull().sum()
#####the dataset has no null values
####there is no  unimportant columns so no need to drop the column

####relationship analysis
correlation=data.corr()
sns.heatmap(correlation,xticklabels=correlation.columns,yticklabels=correlation.columns)

data.mean
data.mode
data.median

ds_cat = data.select_dtypes(include = 'object').copy()
ds_cat.head(2)

sns.countplot(data = ds_cat, x ='Stage')
sns.countplot(data=data,x='age')


data.var() # variance

sns.boxplot(data = data, x='age', y='Nodalstatus')
map(sns.distplot,"NS1")
sns.countplot(data = ds_cat, x = 'Menaupausalstatus')


data['Histology'].unique()



sns.pairplot(data)

########let's add our dependent variable to this dataset

ds_cat['age'] = data.loc[ds_cat.index, 'age'].copy()
sns.boxplot(data =ds_cat, x='Chemo', y='age')

####Let's stack 3 variables (6 charts) and see how it looks like

fig = plt.figure(figsize = (15,10))
ax1 = fig.add_subplot(2,3,1)
sns.countplot(data = ds_cat, x = 'Stage', ax=ax1)
ax2 = fig.add_subplot(2,3,2)
sns.countplot(data = ds_cat, x = 'LVI', ax=ax2)
ax3 = fig.add_subplot(2,3,3)
sns.countplot(data = ds_cat, x = 'AR', ax=ax3)
ax4 = fig.add_subplot(2,3,4)
sns.boxplot(data = ds_cat, x = 'Margins', y = 'age' , ax=ax4)
ax5 = fig.add_subplot(2,3,5)
sns.boxplot(data = ds_cat, x = 'AGE1', y = 'age', ax=ax5)
ax6 = fig.add_subplot(2,3,6)
sns.boxplot(data = ds_cat, x = 'NS1', y = 'age', ax=ax6)


























