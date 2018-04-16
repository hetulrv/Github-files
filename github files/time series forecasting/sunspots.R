library(forecast)
fit5 <- auto.arima(sunspots)
fit5
plot(fit5)
forecast(fit5,3)
accuracy(fit5)
plot(forecast(fit5,15),main = "sunspots data", ylab = "spots",xlab = "time")
