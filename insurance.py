### Import the necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

### Library to encode categorical variable
from sklearn.preprocessing import LabelEncoder

### Linear Regression model libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


### Import the data as a panda dataframe
df = pd.read_csv('D:/Kapil/Python/EDA/insurance premium/insurance.csv')

### Data summary
df.head(10)
df.shape
df.dtypes
df.children = df.children.astype(object)

### Check whether there is any missing values
df.isnull().sum()

### Descriptive Statistics
df.describe()
df[df['charges'] > 45000]['charges'].count()/df['charges'].count()

df = df[df['charges'] <= 45000]

### Data Visualization
### Numerical variable visualization
sns.pairplot(df)

### Numerical and categorical variable visualization
sns.pairplot(df, hue = 'sex')
sns.pairplot(df, hue = 'region')
sns.pairplot(df, hue = 'children')
sns.pairplot(df, hue = 'smoker')

### Correlation of numreic variable
correlations = df.corr(method='pearson')
print(correlations)
f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax = sns.heatmap(correlations, annot=True, cmap='cool')


##Converting objects labels into categorical
df[['sex', 'smoker', 'region', 'children']] = df[['sex', 'smoker', 'region', 'children']].astype('category')
df.dtypes

##Converting category labels into numerical using LabelEncoder
label = LabelEncoder()

label.fit(df.sex.drop_duplicates())
df.sex = label.transform(df.sex)

label.fit(df.smoker.drop_duplicates())
df.smoker = label.transform(df.smoker)

label.fit(df.region.drop_duplicates())
df.region = label.transform(df.region)

label.fit(df.children.drop_duplicates())
df.children = label.transform(df.children)

df.dtypes


### Segregrating dataframe in independent and target variable
x = df.drop(['charges'], axis = 1)  #Independent Variables
y = df['charges']                   #Target Variables

### Prepare a list of models
models = []
models.append(('Linear', LinearRegression()))
models.append(('Ridge', Ridge(alpha = 0.5)))
models.append(('Lasso', Lasso(alpha=0.2, fit_intercept=True, normalize=False, precompute=False, max_iter=1000,
              tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')))
models.append(('Elastic', ElasticNet()))
models.append(('Kneighbour', KNeighborsRegressor()))
models.append(('CART',  DecisionTreeRegressor()))
models.append(('SVR',  SVR()))


### Evaluate each model in terms of R square
results = []
names = []
scoring = 'accuracy'

for name, model in models:
    kfold = KFold(n_splits=10, random_state=7, shuffle = True)
    cv_results = cross_val_score(model, x, y)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Visual representation of model comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()






















