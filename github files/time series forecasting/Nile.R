N<- Nile
#install.packages("tseries")
# there is no need for the transformation as there is a constant variance.
# diff done to make the data stationary. Removes the downward trend
plot(diff(Nile))
library(forecast)
library(tseries)
plot(Nile)
ndiffs(Nile)
dnile <- diff(Nile, d = 1)
plot(dnile)
#Augmented Dickey Fuller Test
#adf.test is used to test whether the time series is stationary or not
adf.test(dnile)
#adf.test(Nile)
#dnile looks more stationary then the Nile dataset
par(mfrow = c(1,1))
Acf(dnile)
#from the plot it is seen that there is a huge autocorrelation at lag 1.
#Hence trying the model ARIMA(0,1,1) looks like a good option as it will provide 
#weights in the increasing order
Pacf(dnile)
#goal is to identify p, d, f values. We already know that d = 1
fit4<- arima(Nile, order = c(0,1,1))
fit4
accuracy(fit4)

qqnorm(fit4$residuals)
qqline(fit4$residuals)
#The Box.test() function provides a test that the autocorrelations are all zero. The
#results aren’t significant, suggesting that the autocorrelations don’t differ from zero.
#This ARIMA model appears to fit the data well.
Box.test(fit4$residuals,type = "Ljung-Box")
forecast(fit4,2)
plot(forecast(fit4,1), xlab = "Year",ylab = "Annual Flow")
