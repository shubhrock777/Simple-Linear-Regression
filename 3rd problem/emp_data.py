import pandas as pd 


#loading the dataset
churn = pd.read_csv("D:/BLR10AM/Assi/20. Simple liner regression/Datasets_SLR/emp_data.csv") 

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary
description  = ["data regarding the employee’s salary, important data",
                "employee churn, important data"]

d_types =["Ratio","Count"]

data_details =pd.DataFrame({"column name":churn.columns,
                            "data types ":d_types,
                            "description":description,
                            "data type(in Python)": churn.dtypes})

            #3.	Data Pre-processing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of churn 
churn.info()
churn.describe()          


#data types        
churn.dtypes


#checking for na value
churn.isna().sum()
churn.isnull().sum()

#checking unique value for each columns
churn.nunique()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    


EDA ={"column ": churn.columns,
      "mean": churn.mean(),
      "median":churn.median(),
      "mode":churn.mode(),
      "standard deviation": churn.std(),
      "variance":churn.var(),
      "skewness":churn.skew(),
      "kurtosis":churn.kurt()}

EDA

# covariance for data set 
covariance = churn.cov()
covariance


####### graphichurn repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(churn.iloc[:, :])


#boxplot for every columns
churn.columns
churn.nunique()

churn.boxplot(column=['Salary_hike', 'Churn_out_rate'])   #no outlier



"""
5.	Model Building:
5.1	Perform Simple Linear Regression on the given datasets
5.2	Apply different transformations such as exponential, log, polynomial transformations and churnculate RMSE values, R-Squared values, Correlation Coefficient for each model
5.3	Build the models and choose the best fit model
5.4	Briefly explain the model output in the documentation	 
 """

import numpy as np
#model bulding 
# Linear Regression model
Co_coe_val_1  = np.corrcoef(churn.Churn_out_rate, churn.Salary_hike)
Co_coe_val_1
# Import library
import statsmodels.formula.api as smf


model1= smf.ols('Churn_out_rate ~ Salary_hike' , data = churn).fit()
model1.summary()

#perdicting on whole data
pred = model1.predict(pd.DataFrame(churn['Salary_hike']))

import matplotlib.pyplot as plt

# Regression Line
plt.scatter(churn.Salary_hike, churn.Churn_out_rate)
plt.plot(churn.Salary_hike, pred, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error churnculation
rmse =  np.sqrt(((pred-churn['Churn_out_rate'])**2).mean())
rmse

#model 2

# Transformation Techniques
# Log transformation applied on 'x'
# input = log(x); output = y
######### Model building on Transformed Data
# Log Transformation

plt.scatter(x = np.log(churn['Churn_out_rate']), y = churn["Salary_hike"], color = 'brown')
Co_coe_val_2  =np.corrcoef(np.log(churn.Churn_out_rate), churn.Salary_hike) #correlation
Co_coe_val_2

model2 = smf.ols('Churn_out_rate ~ np.log(Salary_hike)', data =churn).fit()
model2.summary()


pred2 = model2.predict(pd.DataFrame(churn["Salary_hike"]))

# Regression Line

plt.scatter(np.log(churn.Salary_hike),churn.Churn_out_rate)
plt.plot(np.log(churn.Salary_hike), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error churnculation
rmse2 =  np.sqrt(((pred2-churn['Churn_out_rate'])**2).mean()) 
rmse2


#model3

# Log transformation applied on 'y'
# input = x; output = log(y) 
#### Exponential transformation 
# x = waist; y = log(at) 

plt.scatter(x = churn['Churn_out_rate'], y = np.log(churn['Salary_hike']), color = 'orange') 
Co_coe_val_3  =    np.corrcoef(churn.Churn_out_rate, np.log(churn['Salary_hike'])) #correlation
Co_coe_val_3  



model3 = smf.ols('np.log(churn.Churn_out_rate) ~ churn.Salary_hike ', data = churn).fit() 
model3.summary() 


pred3 = model3.predict(pd.DataFrame(churn['Salary_hike']))
pred3_at = np.exp(pred3) 
pred3_at 

# Regression Line

plt.scatter(churn['Salary_hike'], np.log(churn['Churn_out_rate'])) 
plt.plot(churn['Salary_hike'], pred3, "r") 
plt.legend(['Predicted line', 'Observed data']) 
plt.show()

# Error churnculation
rmse3 =  np.sqrt(((pred3_at-churn['Churn_out_rate'])**2).mean())
rmse3   


##model4

#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

plt.scatter(x = np.log(churn.Churn_out_rate), y = churn.Salary_hike, color = 'orange') 
Co_coe_val_4  =    np.corrcoef( np.log(churn['Churn_out_rate']), churn.Salary_hike) #correlation
Co_coe_val_4

model4 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = churn).fit() 
model4.summary() 

pred4 = model4.predict(pd.DataFrame(churn.Salary_hike)) 
pred4_at = np.exp(pred4) 
pred4_at  


# Regression line
from sklearn.preprocessing import PolynomialFeatures  
poly_reg = PolynomialFeatures(degree = 2)  
X = churn.iloc[:, 0:1].values  
X_poly = poly_reg.fit_transform(X) 
# y = wcat.iloc[:, 1].values

plt.scatter(churn.Salary_hike, np.log(churn.Churn_out_rate)) 
plt.plot(churn['Salary_hike'], pred4, color = 'red') 
plt.legend(['Predicted line', 'Observed data']) 
plt.show() 



# Error churnculation
rmse4 =  np.sqrt(((pred4_at -churn['Churn_out_rate'])**2).mean()) 
rmse4



# Choose the best model using RMSE
Model_details = pd.DataFrame({"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), 
        "RMSE":pd.Series([rmse, rmse2, rmse3, rmse4]),
        "R-squared": pd.Series([model1.rsquared,model2.rsquared,model3.rsquared,model4.rsquared]),
        "Adj. R-squared" : pd.Series([model1.rsquared_adj,model2.rsquared_adj,model3.rsquared_adj,model4.rsquared_adj]),
         "Correlation coefficient values ":pd.Series([Co_coe_val_1,Co_coe_val_2,Co_coe_val_3,Co_coe_val_4])})
         
Model_details

###################
# The best model


from sklearn.model_selection import train_test_split

train , test = train_test_split(churn, test_size = 0.5 , random_state = 775)

#final model
finalmodel = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike) ', data = train).fit()
finalmodel.summary()


# Predict on test data
test_pred_exp = finalmodel.predict(pd.DataFrame(test))
test_pred= np.exp(test_pred_exp)
test_pred


# Model Evaluation on Test data
test_rmse = np.sqrt(((test_pred-test.Churn_out_rate)**2).mean())
test_rmse

# Prediction on train data
train_pred_exp = finalmodel.predict(pd.DataFrame(train))
train_pred= np.exp(train_pred_exp)
train_pred


# Model Evaluation on train data
train_rmse = np.sqrt(((train_pred-train.Churn_out_rate)**2).mean())
train_rmse
