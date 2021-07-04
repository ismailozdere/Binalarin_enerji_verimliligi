# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:34:00 2021

@author: ismail özdere
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot  as plt 
from sklearn.metrics import r2_score


veriler = pd.read_csv("ENB2012_data.csv")



x = veriler.iloc[ : ,0 : 8].values
y = veriler.iloc[:,-2:].values



from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test =  train_test_split(x , y ,
                                                        test_size = 0.33 ,
                                                        random_state = 0)
    

# linear regression 
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x_train , y_train)

lin_ac = r2_score(y_test , lin_reg.predict(x_test))

import statsmodels.api as sm

#X = np.append(arr= np.ones((768 , 1)).astype(int) , 
#              values=veriler.iloc[:,:-1] ,
#              axis=1)

X_l = veriler.iloc[:,[0,1,2,3,4,5,6,7,8,9]].values
X_l = np.array(X_l , dtype=float)
model = sm.OLS(veriler.iloc[:,-2:-1] , X_l).fit()
print(model.summary())


# polynomial regression 
from sklearn.preprocessing import PolynomialFeatures



#degree 2 accuracy
poly_reg2 = PolynomialFeatures(degree= 2)

x_poly2 = poly_reg2.fit_transform(x_train)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2, y_train)

poly_ac2 = r2_score(y_test ,lin_reg2.predict(poly_reg2.fit_transform(x_test)))


#degree 3 accuracy
poly_reg3 = PolynomialFeatures(degree= 3)

x_poly3 = poly_reg3.fit_transform(x_train)

lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3, y_train)

poly_ac3 = r2_score(y_test ,lin_reg3.predict(poly_reg3.fit_transform(x_test)))


#degree 4 accuracy
poly_reg4 = PolynomialFeatures(degree= 4)

x_poly4 = poly_reg4.fit_transform(x_train)

lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly4, y_train)

poly_ac4 = r2_score(y_test ,lin_reg4.predict(poly_reg4.fit_transform(x_test)))


# degree 5 fit 
poly_reg = PolynomialFeatures(degree= 5)

x_poly = poly_reg.fit_transform(x_train)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y_train)


poly_ac5 = r2_score(y_test ,lin_reg2.predict(poly_reg.fit_transform(x_test)))


#degree 6 accuracy
poly_reg6 = PolynomialFeatures(degree= 6)

x_poly6 = poly_reg6.fit_transform(x_train)

lin_reg6 = LinearRegression()
lin_reg6.fit(x_poly6, y_train)

poly_ac6 = r2_score(y_test ,lin_reg6.predict(poly_reg6.fit_transform(x_test)))


# decision tree regression 
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state = 0 )

r_dt.fit(x_train , y_train )
z = x_train + 0.4
k = x_train - 0.4

#dt_pred_z = r_dt.predict(z)
#dt_pred_k = r_dt.predict(k)

de_acc_z = r2_score(y_test ,  r_dt.predict(x_test))
#de_acc_k = r2_score(y_test , dt_pred_k)

#random forest regression 
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10, 
                               random_state=0)

rf_reg.fit(x_train ,y_train )

rf_pred =  rf_reg.predict(x_test)


rf_acc = r2_score(y_test , rf_reg.predict(x_test))


# modellerin değerlendirilmesi en kötüden iyiye doğru şöyledir 
# linear regression       -> 0.8981044566991452
# decision tree           -> 0.9639326414941569
# polinıomiald degree = 6 -> 0.9691239019916584
# random forest           -> 0.9748639792515481
# polinıomiald degree = 5 -> 0.9751964802633196
# polinıomiald degree = 3 -> 0.9796784469696516
# polinıomiald degree = 2 -> 0.9806197900436017
# polinıomiald degree = 4 -> 0.9825487132831329















