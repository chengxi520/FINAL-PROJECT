#install.packages("Amelia")
#install.packages("data.table")
#install.packages("forecast")
#install.packages("gridExtra")
#install.packages("ggpubr")
#install.packages("grid")
#install.packages("tseries")
#install.packages("sarima")
#install.packages("itsadug")
#install.packages("lmtest")

library(Amelia)
library(ggplot2)
library(scales)
library(data.table)
library(forecast)
library(gridExtra)
library(ggpubr)
library(grid)
library(tseries)
library(sarima)
library(itsadug)
library(lmtest)


monthly_sales <- read.csv("DAUTONSA.csv",header = TRUE, stringsAsFactors = FALSE)
#View(monthly_sales)

#The dataset looks like
head(monthly_sales)
#We will be using the monthly domestic auto sales(in thousands) for building our time series

#Before we proceed, let’s check if we have any missing data. Amelia library gives us a missmap function that shows the missing details in a visual map.

missmap(monthly_sales,main = "Missing Values",col = c('blue','yellow'),x.cex = 1.5)

#Since no missing data is found, we can go ahead with visually plotting the daily count to see if we can identify any trends, seasonality, cycle or outlier from the data.

#Before we plot the count data, we need to convert the dteday field from character to date type.

monthly_sales$DATE <- as.Date(monthly_sales$DATE, "%Y-%m-%d")


#We plot the monthly sales data:
uc_ts_plot <- ggplot(monthly_sales, aes(DATE,DAUTONSA)) + geom_line(na.rm=TRUE) + 
  xlab("Month") + ylab("Auto Sales in Thousands") + 
  scale_x_date(labels = date_format(format= "%b-%Y"),breaks = date_breaks("1 year")) +
  stat_smooth(colour = "green")

uc_ts_plot

#There seems to be an outlier that we could see from the plot. This suspected outlier can bias the model by skewing statistical summaries. R provides a convenient method for removing time series outliers: tsclean() as part of its forecast package. tsclean() identifies and replaces outliers using series smoothing and decomposition.

#We need to remove the outlier before we proceed with stationarizing the series. tsclean() is also capable of inputing missing values in the series if there are any. We are using the ts() command to create a time series object to pass to tsclean():

monthly_ts <- ts(monthly_sales[,c('DAUTONSA')])
monthly_sales$csales <- tsclean(monthly_ts)

#Plot the cleaned monthly sales data:
c_ts_plot <- ggplot(monthly_sales, aes(DATE,csales)) + geom_line(na.rm=TRUE) + 
  xlab("Month") + ylab("Auto Sales in Thousands") + 
  scale_x_date(labels = date_format(format= "%b-%Y"),breaks = date_breaks("1 year")) + 
  stat_smooth(colour="green")
c_ts_plot

#Now, let’s compare both cleaned and uncleaned plots:
grid.arrange(uc_ts_plot,c_ts_plot,ncol=1, top = textGrob("Uncleaned vs Cleaned Series"))

##Smoothing the series
#If data points are still volatile, then we can apply smoothing smoothing. By applying smoothing, we can have a better idea about the series and it’s components. It also makes the series more predictable. In this case, we could have used quarterly/biannualy moving average. If the data points are on a daily basis,many level of seasonality(daily, weekly, monthly or yearly) can be incorporated.

#However, looking at the graph, our data does not require any smoothing. Therefore, We go ahead with the cleaned data.
my_ts <- ts(na.omit(monthly_sales$csales),frequency = 12)

#As our data is monthly, we used frequency =12 in above command. We also ignore the NA values.

#Next, we plot the cleaned series to infer visual cues from the graph.

plot(my_ts)

###Identify Level of Differencing Required
#Now that the series is cleaned, we need to remove trend by using appropriate order of difference and make the series stationary. We do this by looking at acf, Dickey-Fuller Test and standard deviation.

##Dickey Fuller test:

#X(t) = Rho * X(t-1) + Er(t) => X(t) - X(t-1) = (Rho - 1) X(t - 1) + Er(t)

#We have to test if Rho - 1 is significantly different than zero or not. If the null hypothesis gets rejected, we’ll get a stationary time series.
#Stationary testing and converting a series into a stationary series are the most critical processes in a time series modelling. We need to memorize each and every detail of this concept to move on to the next step of time series modelling.
#To confirm that the series is not stationary, we perform the augmented Dickey-Fuller Test.

adf.test(my_ts)
#P value is 0.01 indicating the null hypothesis ‘series is non-stationary’ is true i.e the series is not stationary

#Plot the auto-correlation plot for the series to identify the order of differencing required.
Acf(my_ts)

#ACF plot shows positive correlation at higher lags. this indicates that we need differencing to make the series stationary.
#Let’s try order 1 difference
#We will fit ARIMA(0,d,0)(0,D,0)[12] models and verify acf residuals to find which ‘d’ or ‘D’ order of differencing is appropriate in our case.

#Applying only one order of difference i.e ARIMA(0,1,0)(0,0,0)

dfit1 <- arima(my_ts,order = c(0,1,0))
plot(residuals(dfit1))
#
Acf(residuals(dfit1))
#
Pacf(residuals(dfit1))

#The differenced series still shows some strong autocorrelation at the seasonal period 12 and 24. Because the seasonal pattern is strong and stable, we know that we will want to use an order of seasonal differencing in the model.

#Before that let’s try only with one seasonal difference i.e ARIMA(0,0,0)(0,1,0)

dfit2 <- arima(my_ts, order =c(0,0,0), seasonal = list(order = c(0,1,0), period = 12))
plot(residuals(dfit2))

#Looking a the residual plot, residuals does not look like white noise. We also need to check the acf and pacf plots of residuals.
Acf(residuals(dfit2))

Pacf(residuals(dfit2))
#The seasonally differenced series shows a very strong pattern of positive autocorrelation and is similar to a seasonal random walk model. The correlation plots indicate an AR signature and/or incorporating another order of difference into the model.

#Let’s go ahead and apply both seasonal and non-seasonal differencing i,e ARIMA(0,1,0)(0,1,0)[12]

dfit3 <- arima(my_ts, order =c(0,1,0), seasonal = list(order = c(0,1,0), period = 12))
#Next, we check the residuals
plot(residuals(dfit3))

#Residuals seems to return to the mean and we don’t see any pattern in the residuals.

#Below is the acf and Pacf plot of residuals.
Acf(residuals(dfit3))

Pacf(residuals(dfit3))

#ACF at lag 1 is -ve and slightly smaller than -0.2. We know that if the lag 1 acf falls below -0.5, then the series is over differenced. Positive spikes in acf have become negative, another sign of possible over differencing. Therefore, this model might be suffering from slight over differencing. This overdifferencing can be compensated by adding a MA term.

#To select the appropriate order of differencing, we have to consider the error statistics, the standard deviation in specific.

#In below summary, SD is same as RMSE.

arima(x = my_ts, order = c(0, 1, 0))
accuracy(arima(x = my_ts, order = c(0, 1, 0)))


arima(x = my_ts, order = c(0, 0, 0), seasonal = list(order = c(0, 1, 0), period = 12))
accuracy(arima(x = my_ts, order = c(0, 0, 0), seasonal = list(order = c(0, 1, 0), period = 12)))

arima(x = my_ts, order = c(1, 1, 1), seasonal = list(order = c(1, 1, 1), period = 12))
accuracy(arima(x = my_ts, order = c(1, 1, 1), seasonal = list(order = c(1, 1, 1), period = 12)))


##Selecting appropriate order of differencing:
#The optimal order of differencing is often the order of differencing at which the standard deviation is lowest. (Not always, though. Slightly too much or slightly too little differencing can also be corrected with AR or MA terms.

#Out of the above, dfit3 model i.e ARIMA(0,1,0)(0,1,0)12 has the lowest standard deviation(RMSE) and AIC. Therefore, it is the correct order of differencing.

#Therefore, the value of d=1 and D=1 is set. Now, we need to identify AR/MA and SAR/SMA values and fit the model.


###Identifying the AR/MA(p/q) and SAR/SMA(P/Q) components.

#Looking back at the correlation plot of model dfit3, ACF is negative at lag 1 and shows sharp cut-off immediately after lag 1, we can add a MA to the model to compensate for the overdifferencing.

#Since, we do not see any correlation at lag s,2s,3s etc i.e 12,24,36 etc, we do not need to add SAR/SMA to our model.

dfit4 <- arima(my_ts, order =c(1,1,2), seasonal = list(order = c(3,0,0), period = 12), method="CSS")
plot(residuals(dfit4))

Acf(residuals(dfit4))

Pacf(residuals(dfit4))




#Looking at he residual plot for above model, slight amount of correlation remains at lag 10 and 22, but the overall plots seem good.

#Thus, this model seems like a good fit pending statistically significant MA co-efficient and low AIC.

##check for Statistical Significance

#Let’s check the model parameter’s significance. The coeftest() function in lmtest package can help us in getting the p-values of coefficients.
coeftest(dfit4)
#As we can see, P value is negligible and thus the test confirms that MA1 coefficient is statistically significant.


dfit5 <- auto.arima(my_ts, seasonal = TRUE)

plot(residuals(dfit5))

#Residual plot is similar to that of the model we built above.

#Next, we see the acf, pacf and summary of the auto built model.

Acf(residuals(dfit5))

Pacf(residuals(dfit5))


coeftest(dfit5)

#auto arima gives us ARIMA(1,1,2)(1,2,0)[12]
#Auto arima gives us ARIMA(1,1,2)(1,2,0)[12]. All coefficients are significant.

#Clearly this model performs worse than the model we built earlier as ARIMA(1,1,2)(3,0,0)[12] 

#By rule of parsimony and/or minimum AIC, we can reject ARIMA(1,1,2)(1,2,0)[12] and accept ARIMA(1,1,2)(3,0,0)[12] as our model.

###Model Validation
#To see how our model will perform in future, we can use n-fold holdout method.

hold <- window(ts(my_ts), start = 72)

#Fit the model to predict for observation(months) 72 through 83.
fit_predicted <- arima(ts(my_ts[-c(600:616)]), order =c(1,1,2), seasonal = list(order = c(3,0,0), period = 12),method="CSS")

#Use the above model to forecast values for last 10 months. Forecasting using a fitted model is straightforward in R. We can specify forecast horizon h periods ahead for predictions to be made, and use the fitted model to generate those predictions:
forecast_pred <- forecast(fit_predicted,h=15)
plot(forecast_pred)
plot(my_ts)

#In the above graph, blue line is the predicted data and the confidence bands are in dark grey(80%) and light grey(95%).

#Model’s prediction is pretty good and we can see predicted sales closely follow the actual data. This is an indication of a good model.


###Sales Prediction for 2018/19.
#Next step in our model is to forecast values i.e the monthly sales data. We have specified h=24 to predict for next 24 observations(months) i.e next 2 years - 2018 and 2019.
f_values <- forecast(dfit4, h=24)
plot(f_values)

##So, from the forecasting model, we can see that the auto sales seems like is gonna keep going down in the next 2 years.







