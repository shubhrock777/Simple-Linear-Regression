import pandas as pd


#loading the dataset
sat = pd.read_csv("D:/BLR10AM/Assi/20. Simple liner regression/Datasets_SLR/SAT_GPA.csv")

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary
description  = ["SAT scores based on the exam giver’s GPA ",
                "GPA, or Grade Point Average, is a number that indicates how well or how high you scored in your courses on average"]

d_types =["Ratio","Ratio"]

data_details =pd.DataFrame({"column name":sat.columns,
                            "data types ":d_types,
                            "description":description,
                            "data type(in Python)": sat.dtypes})

            #3.	Data Pre-processing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of sat 
sat.info()
sat.describe()          


#data types        
sat.dtypes


#checking for na value
sat.isna().sum()
sat.isnull().sum()

#checking unique value for each columns
sat.nunique()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    


EDA ={"column ": sat.columns,
      "mean": sat.mean(),
      "median":sat.median(),
      "mode":sat.mode(),
      "standard deviation": sat.std(),
      "variance":sat.var(),
      "skewness":sat.skew(),
      "kurtosis":sat.kurt()}

EDA

# covariance for data set 
covariance = sat.cov()
covariance


####### graphisat repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(sat.iloc[:, :])


#boxplot for every columns
sat.columns
sat.nunique()

sat.boxplot(column=['SAT_Scores', 'GPA'])   #no outlier



"""
5.	Model Building:
5.1	Perform Simple Linear Regression on the given datasets
5.2	Apply different transformations such as exponential, log, polynomial transformations and satculate RMSE values, R-Squared values, Correlation Coefficient for each model
5.3	Build the models and choose the best fit model
5.4	Briefly explain the model output in the documentation	 
 """

import numpy as np
#model bulding 
# Linear Regression model
Co_coe_val_1  = np.corrcoef(sat.SAT_Scores, sat.GPA)
Co_coe_val_1
# Import library
import statsmodels.formula.api as smf


model1= smf.ols('SAT_Scores ~ GPA' , data = sat).fit()
model1.summary()

#perdicting on whole data
pred = model1.predict(pd.DataFrame(sat['GPA']))

import matplotlib.pyplot as plt

# Regression Line
plt.scatter(sat.GPA, sat.SAT_Scores)
plt.plot(sat.GPA, pred, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error satculation
rmse =  np.sqrt(((pred-sat['SAT_Scores'])**2).mean())
rmse

#model 2

# Transformation Techniques
# Log transformation applied on 'x'
# input = log(x); output = y
######### Model building on Transformed Data
# Log Transformation

plt.scatter(x = np.log(sat['SAT_Scores']), y = sat["GPA"], color = 'brown')
Co_coe_val_2  =np.corrcoef(np.log(sat.SAT_Scores), sat.GPA) #correlation
Co_coe_val_2

model2 = smf.ols('SAT_Scores ~ np.log(GPA)', data =sat).fit()
model2.summary()


pred2 = model2.predict(pd.DataFrame(sat["GPA"]))

# Regression Line

plt.scatter(np.log(sat.GPA),sat.SAT_Scores)
plt.plot(np.log(sat.GPA), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error satculation
rmse2 =  np.sqrt(((pred2-sat['SAT_Scores'])**2).mean()) 
rmse2


#model3

# Log transformation applied on 'y'
# input = x; output = log(y) 
#### Exponential transformation 
# x = waist; y = log(at) 

plt.scatter(x = sat['SAT_Scores'], y = np.log(sat['GPA']), color = 'orange') 
Co_coe_val_3  =    np.corrcoef(sat.SAT_Scores, np.log(sat['GPA'])) #correlation
Co_coe_val_3  



model3 = smf.ols('np.log(sat.SAT_Scores) ~ sat.GPA ', data = sat).fit() 
model3.summary() 


pred3 = model3.predict(pd.DataFrame(sat['GPA']))
pred3_at = np.exp(pred3) 
pred3_at 

# Regression Line

plt.scatter(sat['GPA'], np.log(sat['SAT_Scores'])) 
plt.plot(sat['GPA'], pred3, "r") 
plt.legend(['Predicted line', 'Observed data']) 
plt.show()

# Error satculation
rmse3 =  np.sqrt(((pred3_at-sat['SAT_Scores'])**2).mean())
rmse3   


##model4

#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

plt.scatter(x = np.log(sat.SAT_Scores), y = sat.GPA, color = 'orange') 
Co_coe_val_4  =    np.corrcoef( np.log(sat['SAT_Scores']), sat.GPA) #correlation
Co_coe_val_4

model4 = smf.ols('np.log(SAT_Scores) ~ GPA + I(GPA*GPA)', data = sat).fit() 
model4.summary() 

pred4 = model4.predict(pd.DataFrame(sat.GPA)) 
pred4_at = np.exp(pred4) 
pred4_at  


# Regression line
from sklearn.preprocessing import PolynomialFeatures  
poly_reg = PolynomialFeatures(degree = 2)  
X = sat.iloc[:, 0:1].values  
X_poly = poly_reg.fit_transform(X) 
# y = wcat.iloc[:, 1].values

plt.scatter(sat.GPA, np.log(sat.SAT_Scores)) 
plt.plot(sat['GPA'], pred4, color = 'red') 
plt.legend(['Predicted line', 'Observed data']) 
plt.show() 



# Error satculation
rmse4 =  np.sqrt(((pred4_at -sat['SAT_Scores'])**2).mean()) 
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

train , test = train_test_split(sat, test_size = 0.7 , random_state = 7)

finalmodel = smf.ols('SAT_Scores ~ GPA', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))


# Model Evaluation on Test data
test_rmse = np.sqrt(((test_pred-test.SAT_Scores)**2).mean())
test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))


# Model Evaluation on train data
train_rmse = np.sqrt(((train_pred-train.SAT_Scores)**2).mean())
train_rmse