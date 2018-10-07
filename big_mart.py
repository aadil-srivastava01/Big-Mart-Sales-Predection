# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#####importing data##########################
df_train = pd.read_csv('train.csv')
df = pd.read_csv('test.csv')

X = df_train.iloc[:,1:-1].values
y = df_train.iloc[:,-1].values#.reshape(-1,1)

X_pred = df.iloc[:,1:11].values

#############encoding data#######################
enc_X = LabelEncoder() 

X[:,1]      = enc_X.fit_transform(X[:,1])
X_pred[:,1] = enc_X.transform(X_pred[:,1])

X[:,3]      = enc_X.fit_transform(X[:,3])
X_pred[:,3] = enc_X.transform(X_pred[:,3])

X[:,5]      = enc_X.fit_transform(X[:,5])
X_pred[:,5] = enc_X.transform(X_pred[:,5])

X[:,7]      = enc_X.fit_transform(X[:,7])
X_pred[:,7] = enc_X.transform(X_pred[:,7])

X[:,8]      = enc_X.fit_transform(X[:,8])
X_pred[:,8] = enc_X.transform(X_pred[:,8])

X[:,9]      = enc_X.fit_transform(X[:,9])
X_pred[:,9] = enc_X.transform(X_pred[:,9])

X = X.astype(float)
X_pred = X_pred.astype(float)

#########feature selection#####################
import statsmodels.formula.api as sm

X= np.append(np.ones((X.shape[0],1),int),X,1)

X_pred = np.append(np.ones((X_pred.shape[0],1),int),X_pred,1)

X_model = X[:,[0,1,2,3,4,5,6,7,8,9,10]]

reg_ols = sm.OLS(endog =y,exog=X_model).fit()
reg_ols.summary()

X_model = X[:,[0,1,2,3,4,5,6,8,9,10]]

reg_ols = sm.OLS(endog =y,exog=X_model).fit()
reg_ols.summary()

X_model = X[:,[0,2,3,4,5,6,8,9,10]]

reg_ols = sm.OLS(endog =y,exog=X_model).fit()
reg_ols.summary()

X_model = X[:,[0,2,3,5,6,8,9,10]]

reg_ols = sm.OLS(endog =y,exog=X_model).fit()
reg_ols.summary()

X_model = X[:,[0,3,5,6,8,9,10]]

reg_ols = sm.OLS(endog =y,exog=X_model).fit()
reg_ols.summary()

X_pred = X_pred[:,[0,3,5,6,8,9,10]]

########################################################

#################one hot encoding######################
X_model = X_model[:,1:7]

X_pred = X_pred[:,1:7]

ohe = OneHotEncoder(categorical_features = [2,3,4,5])
X_model = ohe.fit_transform(X_model).toarray()
X_pred  = ohe.transform(X_pred).toarray()
##########scaling############################

sc_X = StandardScaler()

X_model[:,20:22] = sc_X.fit_transform(X_model[:,20:22])

X_pred[:,20:22]  = sc_X.fit_transform(X_pred[:,20:22])

###########splitting the data############################
X_train,X_test,y_train,y_test = train_test_split(
        X_model, y, test_size = 0.2, random_state = 0)
##############Applying SVM model##########################
from sklearn.svm import SVR

svr = SVR(kernel = 'linear',C=27,gamma='auto',epsilon = 1)
svr.fit(X_train,y_train)
svr.score(X_train,y_train)
svr.score(X_test,y_test)

#achieving a prediction accuracy of 55.27% over the test set'''
####applying ramdon forest model###############
from sklearn.ensemble import  RandomForestRegressor

rfr = RandomForestRegressor(n_estimators = 100,min_samples_leaf=50,
                            oob_score = True, n_jobs = -1, random_state = 0)
rfr.fit(X_train,y_train)
rfr.score(X_train,y_train)
rfr.score(X_test,y_test)

#getting an accuracy of 59.37% with this config of random forest.
##############creating submission file###########

y_pred = rfr.predict(X_pred).reshape(-1,1)

df_pred = pd.DataFrame.from_records(y_pred)

df_pred.to_csv('y_pred.csv',sep='\t',float_format='%.3f')
