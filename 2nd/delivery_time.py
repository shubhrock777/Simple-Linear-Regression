import pandas as pd 


#loading the dataset
delivery = pd.read_csv("D:/BLR10AM/Assi/20. Simple liner regression/Datasets_SLR/delivery_time.csv") 

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary
description  = ["delivery time taken, important data",
                "the time sorted by the restaurants, important data"]

d_types =["Ratio","Count"]

data_details =pd.DataFrame({"column name":delivery.columns,
                            "data types ":d_types,
                            "description":description,
                            "data type(in Python)": delivery.dtypes})

            #3.	Data Pre-processing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of delivery 
delivery.info()
delivery.describe()          

#rename the columns
delivery.rename(columns = {'Delivery Time':'Delivery_Time', 'Sorting Time':'Sorting_Time'}, inplace = True) 
#data types        
delivery.dtypes


#checking for na value
delivery.isna().sum()
delivery.isnull().sum()

#checking unique value for each columns
delivery.nunique()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    


EDA ={"column ": delivery.columns,
      "mean": delivery.mean(),
      "median":delivery.median(),
      "mode":delivery.mode(),
      "standard deviation": delivery.std(),
      "variance":delivery.var(),
      "skewness":delivery.skew(),
      "kurtosis":delivery.kurt()}

EDA

# covariance for data set 
covariance = delivery.cov()
covariance


####### graphidelivery repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(delivery.iloc[:, :])


#boxplot for every columns
delivery.columns
delivery.nunique()

delivery.boxplot(column=["Delivery Time"])   #no outlier


"""
5.	Model Building:
5.1	Perform Simple Linear Regression on the given datasets
5.2	Apply different transformations such as exponential, log, polynomial transformations and deliveryculate RMSE values, R-Squared values, Correlation Coefficient for each model
5.3	Build the models and choose the best fit model
5.4	Briefly explain the model output in the documentation	 
 """

import numpy as np

#model bulding 
# Linear Regression model
Co_coe_val_1  = np.corrcoef(delivery.Delivery_Time, delivery.Sorting_Time)
Co_coe_val_1
# Import library
import statsmodels.formula.api as smf


model1= smf.ols('Delivery_Time ~ Sorting_Time' , data = delivery).fit()
model1.summary()

#perdicting on whole data
pred = model1.predict(pd.DataFrame(delivery['Sorting_Time']))

import matplotlib.pyplot as plt

# Regression Line
plt.scatter(delivery.Sorting_Time, delivery.Delivery_Time)
plt.plot(delivery.Sorting_Time, pred, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error deliveryculation
rmse =  np.sqrt(((pred-delivery['Delivery_Time'])**2).mean())
rmse

#model 2

# Transformation Techniques
# Log transformation applied on 'x'
# input = log(x); output = y
######### Model building on Transformed Data
# Log Transformation

plt.scatter(x = np.log(delivery['Delivery_Time']), y = delivery["Sorting_Time"], color = 'brown')
Co_coe_val_2  =np.corrcoef(np.log(delivery.Delivery_Time), delivery.Sorting_Time) #correlation
Co_coe_val_2

model2 = smf.ols('Delivery_Time ~ np.log(Sorting_Time)', data =delivery).fit()
model2.summary()


pred2 = model2.predict(pd.DataFrame(delivery["Sorting_Time"]))

# Regression Line

plt.scatter(np.log(delivery.Sorting_Time),delivery.Delivery_Time)
plt.plot(np.log(delivery.Sorting_Time), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error deliveryculation
rmse2 =  np.sqrt(((pred2-delivery['Delivery_Time'])**2).mean()) 
rmse2


#model3

# Log transformation applied on 'y'
# input = x; output = log(y) 
#### Exponential transformation 
# x = waist; y = log(at) 

plt.scatter(x = delivery['Delivery_Time'], y = np.log(delivery['Sorting_Time']), color = 'orange') 
Co_coe_val_3  =    np.corrcoef(delivery.Delivery_Time, np.log(delivery['Sorting_Time'])) #correlation
Co_coe_val_3  



model3 = smf.ols('np.log(delivery.Delivery_Time) ~ delivery.Sorting_Time ', data = delivery).fit() 
model3.summary() 


pred3 = model3.predict(pd.DataFrame(delivery['Sorting_Time']))
pred3_at = np.exp(pred3) 
pred3_at 

# Regression Line

plt.scatter(delivery['Sorting_Time'], np.log(delivery['Delivery_Time'])) 
plt.plot(delivery['Sorting_Time'], pred3, "r") 
plt.legend(['Predicted line', 'Observed data']) 
plt.show()

# Error deliveryculation
rmse3 =  np.sqrt(((pred3_at-delivery['Delivery_Time'])**2).mean())
rmse3   


##model4

#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

plt.scatter(x = np.log(delivery.Delivery_Time), y = delivery.Sorting_Time, color = 'orange') 
Co_coe_val_4  =    np.corrcoef( np.log(delivery['Delivery_Time']), delivery.Sorting_Time) #correlation
Co_coe_val_4

model4 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time + I(Sorting_Time*Sorting_Time)', data = delivery).fit() 
model4.summary() 

pred4 = model4.predict(pd.DataFrame(delivery.Sorting_Time)) 
pred4_at = np.exp(pred4) 
pred4_at  


# Regression line
from sklearn.preprocessing import PolynomialFeatures  
poly_reg = PolynomialFeatures(degree = 2)  
X = delivery.iloc[:, 0:1].values  
X_poly = poly_reg.fit_transform(X) 
# y = wcat.iloc[:, 1].values

plt.scatter(delivery.Sorting_Time, np.log(delivery.Delivery_Time)) 
plt.plot(delivery['Sorting_Time'], pred4, color = 'red') 
plt.legend(['Predicted line', 'Observed data']) 
plt.show() 



# Error deliveryculation
rmse4 =  np.sqrt(((pred4_at -delivery['Delivery_Time'])**2).mean()) 
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

train , test = train_test_split(delivery, test_size = 0.5 , random_state = 775)

#final model
finalmodel = smf.ols('np.log(Delivery_Time) ~ Sorting_Time + I(Sorting_Time*Sorting_Time) ', data = train).fit()
finalmodel.summary()


# Predict on test data
test_pred_exp = finalmodel.predict(pd.DataFrame(test))
test_pred= np.exp(test_pred_exp)
test_pred


# Model Evaluation on Test data
test_rmse = np.sqrt(((test_pred-test.Delivery_Time)**2).mean())
test_rmse

# Prediction on train data
train_pred_exp = finalmodel.predict(pd.DataFrame(train))
train_pred= np.exp(train_pred_exp)
train_pred


# Model Evaluation on train data
train_rmse = np.sqrt(((train_pred-train.Delivery_Time)**2).mean())
train_rmse
