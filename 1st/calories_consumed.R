

library(readxl)   #invoke library

# Load the data
data_cal_wt <- read.csv(file.choose(), header=TRUE)
View(data_cal_wt)
attach(data_cal_wt)

# Graphical exploration
plot(data_cal_wt)

dotplot(Weight.gained..grams., main = "Dot Plot of Waist Circumferences")
dotplot(Calories.Consumed, main = "Dot Plot of Adipose Tissue Areas")


boxplot(Weight.gained..grams., col = "dodgerblue4")
boxplot(Calories.Consumed, col = "red", horizontal = T)

hist(Weight.gained..grams.)
hist(Calories.Consumed)

# Normal QQ plot
qqnorm(Weight.gained..grams.)
qqline(Weight.gained..grams.)

qqnorm(Calories.Consumed)
qqline(Calories.Consumed)

hist(Weight.gained..grams., prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(Weight.gained..grams.))             # add a density estimate with defaults
lines(density(Weight.gained..grams., adjust = 2), lty = "dotted")   # add another "smoother" density

hist(Calories.Consumed, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(Calories.Consumed))             # add a density estimate with defaults
lines(density(Calories.Consumed, adjust = 2), lty = "dotted")   # add another "smoother" density

# Bivariate analysis
# Scatter plot
plot(Weight.gained..grams., Calories.Consumed, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "Waist Ciscumference", 
     ylab = "Adipose Tissue area", pch = 20)  # plot(x,y)



# Exploratory data analysis
summary(data_cal_wt)


# Covariance
cov(Weight.gained..grams.,Calories.Consumed)

# Correlation Coefficient
cor(Weight.gained..grams.,Calories.Consumed)

#model bulding 
# Linear Regression model

reg <- lm(Weight.gained..grams.~ Calories.Consumed)
summary(reg)

confint(reg,leavel=0.95)

pred<-predict(reg,interval = "predict")
pred <- as.data.frame(pred) # changing into dataframe
View(pred)


# ggplot for adding Regression line for data
library(ggplot2)

ggplot(data = data_cal_wt, aes(x = Calories.Consumed, y = Weight.gained..grams.)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data_cal_wt, aes(x =Calories.Consumed, y = Weight.gained..grams.))

# Evaluation the model 

cor(pred$fit, Weight.gained..grams.)

rmse <- sqrt(mean(reg$residuals^2))
rmse


# Transformation Techniques
# Log transformation applied on 'x'
# input = log(x); output = y

plot(log(Calories.Consumed),Weight.gained..grams.)
cor(log(Calories.Consumed), Weight.gained..grams.)

reg_log <- lm(Weight.gained..grams. ~ log(Calories.Consumed), data = data_cal_wt)
summary(reg_log)

confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)
cor(pred$fit,Weight.gained..grams.)

rmse <- sqrt(mean(reg_log$residuals^2))
rmse


# Log transformation applied on 'y'
# input = x; output = log(y)

plot(Calories.Consumed, log(Weight.gained..grams.))
cor(Calories.Consumed, log(Weight.gained..grams.))

reg_log1 <- lm(log(Weight.gained..grams.) ~ Calories.Consumed, data = data_cal_wt)
summary(reg_log1)

predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))

pred <- exp(predlog)  # Antilog = Exponential function
pred <- as.data.frame(pred)
cor(pred$fit, wc.at$AT)

res_log1 = AT - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse

# transform the variables to check whether the predicted values are better

reg_sqrt <- lm(Weight.gained..grams.~ sqrt(Calories.Consumed))
summary(reg_sqrt)

confint(reg_sqrt,level=0.95)

predict(reg_sqrt,interval="predict")

# transform the variables to check whether the predicted values are better
reg_log1 <- lm(Weight.gained..grams.~ log(Calories.Consumed))
summary(reg_log1)

confint(reg_log1,level=0.95)

predict(reg_log1,interval="predict")




# Result
## Applying transformation is decreasing Multiple R Squared Value. So model doesnot need further transformation, Multiple R-squared:  0.8968
