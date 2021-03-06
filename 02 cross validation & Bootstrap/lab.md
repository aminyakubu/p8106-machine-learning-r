Cross validation & Bootstrap
================
Amin Yakubu
2/11/2019

The Validation Set Approach
---------------------------

We explore the use of the validation set approach in order to estimate the test error rates that result from fitting various linear models on the Auto data set

``` r
library(ISLR)
```

We begin by using the sample() function to split the set of observations into two halves, by selecting a random subset of 196 observations out of sample() the original 392 observations. We refer to these observations as the training set.

``` r
set.seed(1)
train = sample(392, 196)
```

We then use the subset option in lm() to fit a linear regression using only the observations corresponding to the training set

``` r
data("Auto")

lm.fit = lm(mpg ~ horsepower, data = Auto, subset = train )
```

We now use the predict() function to estimate the response for all 392 observations, and we use the mean() function to calculate the MSE of the 196 observations in the validation set. Note that the -train index below selects only the observations that are not in the training set.

``` r
attach(Auto)
mean((mpg - predict(lm.fit, Auto))[-train]^2)
```

    ## [1] 26.14142

Therefore, the estimated test MSE for the linear regression fit is 26.14.

We can use the poly() function to estimate the test error for the quadratic and cubic regressions.

``` r
lm.fit2 = lm(mpg ~ poly(horsepower, 2), data = Auto, subset = train)
mean((mpg - predict(lm.fit2, Auto))[-train]^2)
```

    ## [1] 19.82259

``` r
lm.fit3 = lm(mpg ~ poly(horsepower, 3), data = Auto, subset = train)
mean((mpg - predict(lm.fit3, Auto))[-train]^2)
```

    ## [1] 19.78252

These error rates are 19.82 and 19.78, respectively. If we choose a different training set instead, then we will obtain somewhat different errors on the validation set

``` r
set.seed(2)
train = sample(392, 196)

lm.fit = lm(mpg ~ horsepower, subset = train)
mean((mpg - predict(lm.fit,Auto))[-train]^2)
```

    ## [1] 23.29559

``` r
lm.fit2 = lm(mpg ~ poly(horsepower ,2),data = Auto, subset = train) 
mean((mpg - predict(lm.fit2 ,Auto))[-train]^2)
```

    ## [1] 18.90124

``` r
lm.fit3 = lm(mpg ~ poly(horsepower ,3), data = Auto,subset = train) 
mean((mpg - predict(lm.fit3, Auto))[-train]^2)
```

    ## [1] 19.2574

Using this split of the observations into a training set and a validation set, we find that the validation set error rates for the models with linear, quadratic, and cubic terms are 23.30, 18.90, and 19.26, respectively.

These results are consistent with our previous findings: a model that predicts mpg using a quadratic function of horsepower performs better than a model that involves only a linear function of horsepower, and there is little evidence in favor of a model that uses a cubic function of horsepower.

Leave-One-Out Cross-Validation
------------------------------

The LOOCV estimate can be automatically computed for any generalized linear model using the glm() and cv.glm() functions.

We can use the glm() function to perform logistic regression by passing in the family="binomial" argument. But if we use glm() to fit a model without passing in the family argument, then it performs linear regression, just like the lm() function.

We will perform linear regression using the glm() function rather than the lm() function because the former can be used together with cv.glm(). The cv.glm() function is part of the boot library.

``` r
library(boot)
```

``` r
glm.fit = glm(mpg ~ horsepower, data = Auto)

cv.err = cv.glm(Auto, glm.fit)
cv.err$delta
```

    ## [1] 24.23151 24.23114

The cv.glm() function produces a list with several components. The two numbers in the delta vector contain the cross-validation results.

Below, we discuss a situation in which the two numbers differ. Our cross-validation estimate for the test error is approximately 24.23.

We can repeat this procedure for increasingly complex polynomial fits. To automate the process, we use the for() function to initiate a for loop which iteratively fits polynomial regressions for polynomials of order i = 1 to i = 5, computes the associated cross-validation error, and stores it in the ith element of the vector cv.error

``` r
cv.error = rep(0, 5)

for (i in 1:5) {
  glm.fit = glm(mpg ~ poly(horsepower, i), data = Auto)
  cv.error[i] = cv.glm(Auto, glm.fit)$delta[1]
}

cv.error
```

    ## [1] 24.23151 19.24821 19.33498 19.42443 19.03321

As in Figure 5.4, we see a sharp drop in the estimated test MSE between the linear and quadratic fits, but then no clear improvement from using higher-order polynomials

k-Fold Cross-Validation
-----------------------

The cv.glm() function can also be used to implement k-fold CV. Below we use k = 10, a common choice for k, on the Auto data set.

``` r
set.seed(17)
cv.error.10 = rep(0, 10)

for (i in 1:10) {
  glm.fit = glm(mpg ~ poly(horsepower, i), data = Auto)
  cv.error.10[i] = cv.glm(Auto, glm.fit, K = 10)$delta[1]
}

cv.error.10
```

    ##  [1] 24.20520 19.18924 19.30662 19.33799 18.87911 19.02103 18.89609
    ##  [8] 19.71201 18.95140 19.50196

We still see little evidence that using cubic or higher-order polynomial terms leads to lower test error than simply using a quadratic fit.

We saw in Section 5.3.2 that the two numbers associated with delta are essentially the same when LOOCV is performed. When we instead perform k-fold CV, then the two numbers associated with delta differ slightly. The first is the standard k-fold CV estimate. The second is a bias-corrected version. On this data set, the two estimates are very similar to each other.

Bootstrap
---------

``` r
data("Portfolio")
```

To illustrate the use of the bootstrap on this data, we must first create a function, alpha.fn(), which takes as input the (X,Y) data as well as a vector indicating which observations should be used to estimate α. The function then outputs the estimate for α based on the selected observations.

``` r
alpha.fn = function(data, index){
  X = data$X[index]
  Y = data$Y[index]
   
  return((var(Y) - cov(X,Y)) / (var(X) + var(Y) - 2 * cov(X,Y)))

}
```

``` r
alpha.fn(Portfolio, 1:100)
```

    ## [1] 0.5758321

The next command uses the sample() function to randomly select 100 ob- servations from the range 1 to 100, with replacement. This is equivalent to constructing a new bootstrap data set and recomputing αˆ based on the new data set.

``` r
set.seed(1)
alpha.fn(Portfolio, sample(100, 100,replace = T))
```

    ## [1] 0.5963833

We can implement a bootstrap analysis by performing this command many times, recording all of the corresponding estimates for α, and computing the resulting standard deviation. However, the boot() function automates this approach. Below we produce R = 1, 000 bootstrap estimates for α.

``` r
boot(Portfolio, alpha.fn, R = 1000)
```

    ## 
    ## ORDINARY NONPARAMETRIC BOOTSTRAP
    ## 
    ## 
    ## Call:
    ## boot(data = Portfolio, statistic = alpha.fn, R = 1000)
    ## 
    ## 
    ## Bootstrap Statistics :
    ##      original        bias    std. error
    ## t1* 0.5758321 -7.315422e-05  0.08861826

The final output shows that using the original data, αˆ = 0.5758, and that the bootstrap estimate for SE(αˆ) is 0.0886.

Estimating the Accuracy of a Linear Regression Model
----------------------------------------------------

Here we use the bootstrap approach in order to assess the variability of the estimates for β0 and β1, the intercept and slope terms for the linear regression model that uses horsepower to predict mpg in the Auto data set. We will compare the estimates obtained using the bootstrap to those obtained using the formulas for SE(βˆ0) and SE(βˆ1) (produced automatically by the regression model)

``` r
boot.fn = function(data,index) {
  
  return(coef(lm(mpg ~ horsepower, data = data, subset = index))) 
}

boot.fn(Auto, 1:392)
```

    ## (Intercept)  horsepower 
    ##  39.9358610  -0.1578447

``` r
set.seed(1)

boot.fn(Auto, sample(392, 392, replace = T)) 
```

    ## (Intercept)  horsepower 
    ##  38.7387134  -0.1481952

``` r
boot.fn(Auto, sample(392, 392, replace = T))
```

    ## (Intercept)  horsepower 
    ##  40.0383086  -0.1596104

Next, we use the boot() function to compute the standard errors of 1,000 bootstrap estimates for the intercept and slope terms.

``` r
boot(Auto, boot.fn, 1000)
```

    ## 
    ## ORDINARY NONPARAMETRIC BOOTSTRAP
    ## 
    ## 
    ## Call:
    ## boot(data = Auto, statistic = boot.fn, R = 1000)
    ## 
    ## 
    ## Bootstrap Statistics :
    ##       original      bias    std. error
    ## t1* 39.9358610  0.02972191 0.860007896
    ## t2* -0.1578447 -0.00030823 0.007404467

This indicates that the bootstrap estimate for SE(βˆ0) is 0.86, and that the bootstrap estimate for SE(βˆ1) is 0.0074. The standard formulas can be used to compute the standard errors for the regression coefficients in a linear model. These can be obtained using the summary() function.

``` r
summary(lm(mpg ~ horsepower, data=Auto))$coef
```

    ##               Estimate  Std. Error   t value      Pr(>|t|)
    ## (Intercept) 39.9358610 0.717498656  55.65984 1.220362e-187
    ## horsepower  -0.1578447 0.006445501 -24.48914  7.031989e-81

The standard error estimates for βˆ0 and βˆ1 obtained using the formulas from Section 3.1.2 are 0.717 for the intercept and 0.0064 for the slope. Interestingly, these are somewhat different from the estimates obtained using the bootstrap. Does this indicate a problem with the bootstrap? In fact, it suggests the opposite. Recall that the standard formulas given in Equation 3.8 on page 66 rely on certain assumptions. For example, they depend on the unknown parameter σ2, the noise variance. We then estimate σ2 using the RSS. Now although the formula for the standard errors do not rely on the linear model being correct, the estimate for σ2 does. We see in Figure 3.8 on page 91 that there is a non-linear relationship in the data, and so the residuals from a linear fit will be inflated, and so will σˆ2. Secondly, the standard formulas assume (somewhat unrealistically) that the xi are fixed, and all the variability comes from the variation in the errors εi. The bootstrap approach does not rely on any of these assumptions, and so it is likely giving a more accurate estimate of the standard errors of βˆ0 and βˆ1 than is the summary() function.

Below we compute the bootstrap standard error estimates and the stan- dard linear regression estimates that result from fitting the quadratic model to the data.

``` r
boot.fn = function(data,index){
  coefficients(lm(mpg ~ horsepower+I(horsepower^2),data = data, subset = index))
}

set.seed(1)
boot(Auto, boot.fn, 1000)
```

    ## 
    ## ORDINARY NONPARAMETRIC BOOTSTRAP
    ## 
    ## 
    ## Call:
    ## boot(data = Auto, statistic = boot.fn, R = 1000)
    ## 
    ## 
    ## Bootstrap Statistics :
    ##         original        bias     std. error
    ## t1* 56.900099702  6.098115e-03 2.0944855842
    ## t2* -0.466189630 -1.777108e-04 0.0334123802
    ## t3*  0.001230536  1.324315e-06 0.0001208339

``` r
summary(lm(mpg ~ horsepower + I(horsepower^2), data = Auto))$coef
```

    ##                     Estimate   Std. Error   t value      Pr(>|t|)
    ## (Intercept)     56.900099702 1.8004268063  31.60367 1.740911e-109
    ## horsepower      -0.466189630 0.0311246171 -14.97816  2.289429e-40
    ## I(horsepower^2)  0.001230536 0.0001220759  10.08009  2.196340e-21

Since this model provides a good fit to the data (Figure 3.8), there is now a better correspondence between the bootstrap estimates and the standard estimates of SE(βˆ0), SE(βˆ1) and SE(βˆ2).
