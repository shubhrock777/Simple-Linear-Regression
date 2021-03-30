import pandas as pd


#loading the dataset
cal = pd.read_csv("D:/BLR10AM/Assi/20. Simple liner regression/Datasets_SLR/calories_consumed.csv")

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary
description  = ["important data weight gained in grams ",
                "Consumptation in Calories"]

d_types =["Ratio","Ratio"]

data_details =pd.DataFrame({"column name":cal.columns,
                            "data types ":d_types,
                            "description":description,
                            "data type(in Python)": cal.dtypes})

            #3.	Data Pre-processing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of cal 
cal.info()
cal.describe()          


#rename the columns
cal.rename(columns = {'Weight gained (grams)':'weight', 'Calories Consumed':'calories'}, inplace = True) 

#data types        
cal.dtypes


#checking for na value
cal.isna().sum()
cal.isnull().sum()

#checking unique value for each columns
cal.nunique()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    


EDA ={"column ": cal.columns,
      "mean": cal.mean(),
      "median":cal.median(),
      "mode":cal.mode(),
      "standard deviation": cal.std(),
      "variance":cal.var(),
      "skewness":cal.skew(),
      "kurtosis":cal.kurt()}

EDA

# covariance for data set 
covariance = cal.cov()
covariance


####### graphical repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(cal.iloc[:, :])


#boxplot for every columns
cal.columns
cal.nunique()

cal.boxplot(column=['weight', 'calories'])   #no outlier



"""
5.	Model Building:
5.1	Perform Simple Linear Regression on the given datasets
5.2	Apply different transformations such as exponential, log, polynomial transformations and calculate RMSE values, R-Squared values, Correlation Coefficient for each model
5.3	Build the models and choose the best fit model
5.4	Briefly explain the model output in the documentation	 
 """


#model bulding 
# Linear Regression model
Co_coe_val_1  =np.corrcoef(cal.calories, cal.weight)
Co_coe_val_1
# Import library
import statsmodels.formula.api as smf


model1= smf.ols('calories ~ weight' , data = cal).fit()
model1.summary()

#perdicting on whole data
pred = model1.predict(pd.DataFrame(cal['weight']))

import matplotlib.pyplot as plt

# Regression Line
plt.scatter(cal.weight, cal.calories)
plt.plot(cal.weight, pred, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

import numpy as np
# Error calculation
rmse =  np.sqrt(((pred-cal['calories'])**2).mean())
rmse

#model 2

# Transformation Techniques
# Log transformation applied on 'x'
# input = log(x); output = y
######### Model building on Transformed Data
# Log Transformation

plt.scatter(x = np.log(cal['calories']), y = cal["weight"], color = 'brown')
Co_coe_val_2  =np.corrcoef(np.log(cal.calories), cal.weight) #correlation
Co_coe_val_2

model2 = smf.ols('calories ~ np.log(weight)', data =cal).fit()
model2.summary()


pred2 = model2.predict(pd.DataFrame(cal["weight"]))

# Regression Line

plt.scatter(np.log(cal.weight),cal.calories)
plt.plot(np.log(cal.weight), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
rmse2 =  np.sqrt(((pred2-cal['calories'])**2).mean()) 
rmse2

# Log transformation applied on 'y'
# input = x; output = log(y) 
#### Exponential transformation 
# x = waist; y = log(at) 

plt.scatter(x = cal['calories'], y = np.log(cal['weight']), color = 'orange') 
Co_coe_val_2  =    np.corrcoef(cal.calories, np.log(cal['weight'])) #correlation
Co_coe_val_2  
#model3

model3 = smf.ols('np.log(cal.calories) ~ cal.weight ', data = cal).fit() 
model3.summary() 


pred3 = model3.predict(pd.DataFrame(cal['weight']))
pred3_at = np.exp(pred3) 
pred3_at 

# Regression Line

plt.scatter(cal['weight'], np.log(cal['calories'])) 
plt.plot(cal['weight'], pred3, "r") 
plt.legend(['Predicted line', 'Observed data']) 
plt.show()

# Error calculation
rmse3 =  np.sqrt(((pred3_at-cal['calories'])**2).mean())
rmse3   



#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(calories) ~ weight + I(weight*weight)', data = cal).fit() 
model4.summary() 


pred4 = model4.predict(pd.DataFrame(cal.weight)) 
pred4_at = np.exp(pred4) 
pred4_at  


# Regression line
from sklearn.preprocessing import PolynomialFeatures  
poly_reg = PolynomialFeatures(degree = 2)  
X = cal.iloc[:, 0:1].values  
X_poly = poly_reg.fit_transform(X) 
# y = wcat.iloc[:, 1].values

plt.scatter(cal.weight, np.log(cal.calories)) 
plt.plot(cal['weight'], pred4, color = 'red') 
plt.legend(['Predicted line', 'Observed data']) 
plt.show() 



# Error calculation
rmse4 =  np.sqrt(((pred4_at -cal['calories'])**2).mean()) 
rmse4



# Choose the best model using RMSE
Model_details = pd.DataFrame({"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), 
        "RMSE":pd.Series([rmse, rmse2, rmse3, rmse4]),
        "R-squared": pd.Series([model1.rsquared,model2.rsquared,model3.rsquared,model4.rsquared]),
        "Adj. R-squared" : pd.Series([model1.rsquared_adj,model2.rsquared_adj,model3.rsquared_adj,model4.rsquared_adj])})

Model_details

###################
# The best model

from sklearn.model_selection import train_test_split

train , test = train_test_split(cal, test_size = 0.5 , random_state = 7)

finalmodel = smf.ols('calories ~ weight', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))


# Model Evaluation on Test data
test_rmse = np.sqrt(((test_pred-test.calories)**2).mean())
test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))


# Model Evaluation on train data
train_rmse = np.sqrt(((train_pred-train.calories)**2).mean())
train_rmse



# Result
## Applying transformation is decreasing Multiple R Squared Value. So model doesnot need further transformation, Multiple R-squared:  0.911
