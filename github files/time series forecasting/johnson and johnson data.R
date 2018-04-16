library(forecast)
fit3 <- ets(JohnsonJohnson)
fit3
pred3 <- forecast(fit3,11)
pred3
plot(pred3, main = "Johnson and Johnson Data", ylab = "Quarterly Earnings(Dollars)", xlab = "Time", flty = 2)
