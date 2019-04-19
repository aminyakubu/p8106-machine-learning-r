Lab: Decision Trees
================

``` r
library(tree)
library(ISLR)
```

Fitting Classification Trees
============================

We will use classification trees to analyze the Carseats data set. In these data, Sales is a continuous variable, and so we begin by recoding it as a binary variable. We create a variable, called High, which takes on a value of Yes if the Sales variable exceeds 8, and takes on a value of No otherwise

``` r
attach(Carseats)
High = ifelse(Sales <= 8, "No", "Yes")
```

Finally, we use the data.frame() function to merge High with the rest of the Carseats data

``` r
Carseats = data.frame(Carseats, High)
```

We now use the tree() function to fit a classification tree in order to predict High using all variables but Sales.The syntax of the tree() function is quite tree() similar to that of the lm() function

``` r
tree.carseats = tree(High ~ . -Sales, Carseats )
```

The summary() function lists the variables that are used as internal nodes in the tree, the number of terminal nodes, and the (training) error rate.

``` r
summary(tree.carseats)
```

    ## 
    ## Classification tree:
    ## tree(formula = High ~ . - Sales, data = Carseats)
    ## Variables actually used in tree construction:
    ## [1] "ShelveLoc"   "Price"       "Income"      "CompPrice"   "Population" 
    ## [6] "Advertising" "Age"         "US"         
    ## Number of terminal nodes:  27 
    ## Residual mean deviance:  0.4575 = 170.7 / 373 
    ## Misclassification error rate: 0.09 = 36 / 400

We see that the training error rate is 9%. For classification trees, the deviance is also reported in the output of summary().

A small deviance indicates a tree that provides a good fit to the (training) data. The residual mean deviance reported is simply the deviance divided by n − |T0|, which in this case is 400−27 = 373.

We use the plot() function to display the tree structure, and the text() function to display the node labels. The argument pretty=0 instructs R to include the category names for any qualitative predictors, rather than simply displaying a letter for each category

``` r
plot(tree.carseats)
text(tree.carseats ,pretty = 0)
```

![](Lab_Decision_Trees_files/figure-markdown_github/unnamed-chunk-6-1.png)

The most important indicator of Sales appears to be shelving location, since the first branch differentiates Good locations from Bad and Medium locations

If we just type the name of the tree object, R prints output corresponding to each branch of the tree. R displays the split criterion (e.g. Price&lt;92.5), the number of observations in that branch, the deviance, the overall prediction for the branch (Yes or No), and the fraction of observations in that branch that take on values of Yes and No. Branches that lead to terminal nodes are indicated using asterisks.

``` r
tree.carseats
```

    ## node), split, n, deviance, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##   1) root 400 541.500 No ( 0.59000 0.41000 )  
    ##     2) ShelveLoc: Bad,Medium 315 390.600 No ( 0.68889 0.31111 )  
    ##       4) Price < 92.5 46  56.530 Yes ( 0.30435 0.69565 )  
    ##         8) Income < 57 10  12.220 No ( 0.70000 0.30000 )  
    ##          16) CompPrice < 110.5 5   0.000 No ( 1.00000 0.00000 ) *
    ##          17) CompPrice > 110.5 5   6.730 Yes ( 0.40000 0.60000 ) *
    ##         9) Income > 57 36  35.470 Yes ( 0.19444 0.80556 )  
    ##          18) Population < 207.5 16  21.170 Yes ( 0.37500 0.62500 ) *
    ##          19) Population > 207.5 20   7.941 Yes ( 0.05000 0.95000 ) *
    ##       5) Price > 92.5 269 299.800 No ( 0.75465 0.24535 )  
    ##        10) Advertising < 13.5 224 213.200 No ( 0.81696 0.18304 )  
    ##          20) CompPrice < 124.5 96  44.890 No ( 0.93750 0.06250 )  
    ##            40) Price < 106.5 38  33.150 No ( 0.84211 0.15789 )  
    ##              80) Population < 177 12  16.300 No ( 0.58333 0.41667 )  
    ##               160) Income < 60.5 6   0.000 No ( 1.00000 0.00000 ) *
    ##               161) Income > 60.5 6   5.407 Yes ( 0.16667 0.83333 ) *
    ##              81) Population > 177 26   8.477 No ( 0.96154 0.03846 ) *
    ##            41) Price > 106.5 58   0.000 No ( 1.00000 0.00000 ) *
    ##          21) CompPrice > 124.5 128 150.200 No ( 0.72656 0.27344 )  
    ##            42) Price < 122.5 51  70.680 Yes ( 0.49020 0.50980 )  
    ##              84) ShelveLoc: Bad 11   6.702 No ( 0.90909 0.09091 ) *
    ##              85) ShelveLoc: Medium 40  52.930 Yes ( 0.37500 0.62500 )  
    ##               170) Price < 109.5 16   7.481 Yes ( 0.06250 0.93750 ) *
    ##               171) Price > 109.5 24  32.600 No ( 0.58333 0.41667 )  
    ##                 342) Age < 49.5 13  16.050 Yes ( 0.30769 0.69231 ) *
    ##                 343) Age > 49.5 11   6.702 No ( 0.90909 0.09091 ) *
    ##            43) Price > 122.5 77  55.540 No ( 0.88312 0.11688 )  
    ##              86) CompPrice < 147.5 58  17.400 No ( 0.96552 0.03448 ) *
    ##              87) CompPrice > 147.5 19  25.010 No ( 0.63158 0.36842 )  
    ##               174) Price < 147 12  16.300 Yes ( 0.41667 0.58333 )  
    ##                 348) CompPrice < 152.5 7   5.742 Yes ( 0.14286 0.85714 ) *
    ##                 349) CompPrice > 152.5 5   5.004 No ( 0.80000 0.20000 ) *
    ##               175) Price > 147 7   0.000 No ( 1.00000 0.00000 ) *
    ##        11) Advertising > 13.5 45  61.830 Yes ( 0.44444 0.55556 )  
    ##          22) Age < 54.5 25  25.020 Yes ( 0.20000 0.80000 )  
    ##            44) CompPrice < 130.5 14  18.250 Yes ( 0.35714 0.64286 )  
    ##              88) Income < 100 9  12.370 No ( 0.55556 0.44444 ) *
    ##              89) Income > 100 5   0.000 Yes ( 0.00000 1.00000 ) *
    ##            45) CompPrice > 130.5 11   0.000 Yes ( 0.00000 1.00000 ) *
    ##          23) Age > 54.5 20  22.490 No ( 0.75000 0.25000 )  
    ##            46) CompPrice < 122.5 10   0.000 No ( 1.00000 0.00000 ) *
    ##            47) CompPrice > 122.5 10  13.860 No ( 0.50000 0.50000 )  
    ##              94) Price < 125 5   0.000 Yes ( 0.00000 1.00000 ) *
    ##              95) Price > 125 5   0.000 No ( 1.00000 0.00000 ) *
    ##     3) ShelveLoc: Good 85  90.330 Yes ( 0.22353 0.77647 )  
    ##       6) Price < 135 68  49.260 Yes ( 0.11765 0.88235 )  
    ##        12) US: No 17  22.070 Yes ( 0.35294 0.64706 )  
    ##          24) Price < 109 8   0.000 Yes ( 0.00000 1.00000 ) *
    ##          25) Price > 109 9  11.460 No ( 0.66667 0.33333 ) *
    ##        13) US: Yes 51  16.880 Yes ( 0.03922 0.96078 ) *
    ##       7) Price > 135 17  22.070 No ( 0.64706 0.35294 )  
    ##        14) Income < 46 6   0.000 No ( 1.00000 0.00000 ) *
    ##        15) Income > 46 11  15.160 Yes ( 0.45455 0.54545 ) *

In order to properly evaluate the performance of a classification tree on these data, we must estimate the test error rather than simply computing the training error. We split the observations into a training set and a test set, build the tree using the training set, and evaluate its performance on the test data. The predict() function can be used for this purpose. In the case of a classification tree, the argument type="class" instructs R to return the actual class prediction. This approach leads to correct predictions for around 71.5 % of the locations in the test data set.

``` r
set.seed(2)
train = sample(1:nrow(Carseats), 200)
Carseats.test = Carseats[-train,]
High.test = High[-train]

tree.carseats = tree(High ~ .-Sales, Carseats, subset = train)

tree.pred = predict(tree.carseats, Carseats.test, type = "class")
table(tree.pred, High.test)
```

    ##          High.test
    ## tree.pred No Yes
    ##       No  86  27
    ##       Yes 30  57

``` r
(86 + 57)/200 
```

    ## [1] 0.715

Next, we consider whether pruning the tree might lead to improved results. The function cv.tree() performs cross validation in order to determine the optimal level of tree complexity; cost complexity pruning is used in order to select a sequence of trees for consideration.

We use the argument FUN=prune.misclass in order to indicate that we want the classification error rate to guide the cross validation and pruning process, rather than the default for the cv.tree() function, which is deviance.

The cv.tree() function reports the number of terminal nodes of each tree considered (size) as well as the corresponding error rate and the value of the cost complexity parameter used (k corresponds to alpha).

``` r
set.seed(3)
cv.carseats = cv.tree(tree.carseats, FUN = prune.misclass)

names(cv.carseats)
```

    ## [1] "size"   "dev"    "k"      "method"

``` r
cv.carseats$size
```

    ## [1] 19 17 14 13  9  7  3  2  1

``` r
cv.carseats$dev
```

    ## [1] 55 55 53 52 50 56 69 65 80

``` r
cv.carseats$k
```

    ## [1]       -Inf  0.0000000  0.6666667  1.0000000  1.7500000  2.0000000
    ## [7]  4.2500000  5.0000000 23.0000000

``` r
cv.carseats$method
```

    ## [1] "misclass"

Note that, despite the name, dev corresponds to the cross validation error rate in this instance. The tree with 9 terminal nodes results in the lowest cross validation error rate, with 50 cross validation errors. We plot the error rate as a function of both size and k.

``` r
par(mfrow = c(1, 2))
plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")
```

![](Lab_Decision_Trees_files/figure-markdown_github/unnamed-chunk-10-1.png)

We now apply the prune.misclass() function in order to prune the tree to obtain the nine node tree

``` r
prune.carseats = prune.misclass(tree.carseats, best = 9)
plot(prune.carseats)
text(prune.carseats, pretty = 0)
```

![](Lab_Decision_Trees_files/figure-markdown_github/unnamed-chunk-11-1.png)

we apply the predict() function to see how well does this pruned tree perform on the test data set.

``` r
tree.pred = predict(prune.carseats, Carseats.test, type = "class")
table(tree.pred, High.test)
```

    ##          High.test
    ## tree.pred No Yes
    ##       No  94  24
    ##       Yes 22  60

``` r
(94 + 60)/200
```

    ## [1] 0.77

Now 77 % of the test observations are correctly classified, so not only has the pruning process produced a more interpretable tree, but it has also improved the classification accuracy. If we increase the value of best, we obtain a larger pruned tree with lower classification accuracy

``` r
prune.carseats = prune.misclass(tree.carseats, best=15)
plot(prune.carseats)
text(prune.carseats, pretty = 0)
```

![](Lab_Decision_Trees_files/figure-markdown_github/unnamed-chunk-13-1.png)

``` r
tree.pred = predict(prune.carseats, Carseats.test, type = "class")
table(tree.pred, High.test)
```

    ##          High.test
    ## tree.pred No Yes
    ##       No  86  22
    ##       Yes 30  62

``` r
(86 + 62)/200
```

    ## [1] 0.74

Fitting Regression Trees
========================

Here we fit a regression tree to the Boston data set. First, we create a training set, and fit the tree to the training data.

``` r
library(MASS)
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)

tree.boston = tree(medv ~ . , Boston, subset = train)
summary(tree.boston)
```

    ## 
    ## Regression tree:
    ## tree(formula = medv ~ ., data = Boston, subset = train)
    ## Variables actually used in tree construction:
    ## [1] "lstat" "rm"    "dis"  
    ## Number of terminal nodes:  8 
    ## Residual mean deviance:  12.65 = 3099 / 245 
    ## Distribution of residuals:
    ##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
    ## -14.10000  -2.04200  -0.05357   0.00000   1.96000  12.60000

Notice that the output of summary() indicates that only three of the variables have been used in constructing the tree. In the context of a regression tree, the deviance is simply the sum of squared errors for the tree. We now plot the tree

``` r
plot(tree.boston)
text(tree.boston, pretty = 0)
```

![](Lab_Decision_Trees_files/figure-markdown_github/unnamed-chunk-15-1.png)

The variable lstat measures the percentage of individuals with lower socioeconomic status. The tree indicates that lower values of lstat correspond to more expensive houses. The tree predicts a median house price of $46, 400 for larger homes in suburbs in which residents have high socioeconomic status (rm&gt;=7.437 and lstat&lt;9.715).

Now we use the cv.tree() function to see whether pruning the tree will improve performance.

``` r
cv.boston = cv.tree(tree.boston)

plot(cv.boston$size, cv.boston$dev, type = 'b')
```

![](Lab_Decision_Trees_files/figure-markdown_github/unnamed-chunk-16-1.png)

In this case, the most complex tree is selected by cross validation. However, if we wish to prune the tree, we could do so as follows, using the prune.tree() function

``` r
prune.boston=prune.tree(tree.boston, best = 5)
plot(prune.boston)
text(prune.boston, pretty = 0)
```

![](Lab_Decision_Trees_files/figure-markdown_github/unnamed-chunk-17-1.png)

In keeping with the cross validation results, we use the unpruned tree to make predictions on the test set.

``` r
yhat = predict(tree.boston, newdata = Boston[-train ,])
boston.test = Boston[-train, "medv"]

plot(yhat, boston.test)
abline(0,1)
```

![](Lab_Decision_Trees_files/figure-markdown_github/unnamed-chunk-18-1.png)

``` r
mean((yhat - boston.test)^2)
```

    ## [1] 25.04559

In other words, the test set MSE associated with the regression tree is 25.05. The square root of the MSE is therefore around 5.005, indicating that this model leads to test predictions that are within around $5, 005 of the true median home value for the suburb.

Bagging and Random Forests
==========================

Here we apply bagging and random forests to the Boston data, using the randomForest package. Bagging is simply a special case of a random forest with m = p. Therefore, the randomForest() function can be used to perform both random forests and bagging. We perform bagging as follows

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
set.seed (1)

bag.boston = randomForest(medv ~ ., data = Boston, subset = train,
                          mtry = 13,
                          importance = TRUE)
bag.boston
```

    ## 
    ## Call:
    ##  randomForest(formula = medv ~ ., data = Boston, mtry = 13, importance = TRUE,      subset = train) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 13
    ## 
    ##           Mean of squared residuals: 11.15723
    ##                     % Var explained: 86.49

The argument mtry=13 indicates that all 13 predictors should be considered for each split of the tree—in other words, that bagging should be done. How well does this bagged model perform on the test set?

``` r
yhat.bag = predict(bag.boston, newdata = Boston[-train,])

plot(yhat.bag, boston.test)
abline(0,1)
```

![](Lab_Decision_Trees_files/figure-markdown_github/unnamed-chunk-20-1.png)

``` r
mean((yhat.bag - boston.test)^2)
```

    ## [1] 13.50808

The test set MSE associated with the bagged regression tree is 13.5, almost half that obtained using an optimally pruned single tree. We could change the number of trees grown by randomForest() using the ntree argument.

``` r
bag.boston=randomForest(medv ~ ., data = Boston, subset = train, 
                        mtry = 13, 
                        ntree = 25)

yhat.bag = predict(bag.boston, newdata = Boston[-train,])

mean((yhat.bag - boston.test)^2) 
```

    ## [1] 13.94835

Growing a random forest proceeds in exactly the same way, except that we use a smaller value of the mtry argument. By default, randomForest() uses p/3 variables when building a random forest of regression trees, and √p variables when building a random forest of classification trees. Here we use mtry = 6.

``` r
set.seed(1)
rf.boston = randomForest(medv ~ ., data = Boston, subset = train,
                       mtry = 6,
                       importance = TRUE)

yhat.rf = predict(rf.boston, newdata = Boston[-train ,])
mean((yhat.rf - boston.test)^2)
```

    ## [1] 11.66454

The test set MSE is 11.66; this indicates that random forests yielded an improvement over bagging in this case.

Using the importance() function, we can view the importance of each variable.

``` r
importance(rf.boston)
```

    ##           %IncMSE IncNodePurity
    ## crim    12.132320     986.50338
    ## zn       1.955579      57.96945
    ## indus    9.069302     882.78261
    ## chas     2.210835      45.22941
    ## nox     11.104823    1044.33776
    ## rm      31.784033    6359.31971
    ## age     10.962684     516.82969
    ## dis     15.015236    1224.11605
    ## rad      4.118011      95.94586
    ## tax      8.587932     502.96719
    ## ptratio 12.503896     830.77523
    ## black    6.702609     341.30361
    ## lstat   30.695224    7505.73936

Two measures of variable importance are reported. The former is based upon the mean decrease of accuracy in predictions on the out of bag samples when a given variable is excluded from the model. The latter is a measure of the total decrease in node impurity that results from splits over that variable, averaged over all trees.

In the case of regression trees, the node impurity is measured by the training RSS, and for classification trees by the deviance. Plots of these importance measures can be produced using the varImpPlot() function.

``` r
varImpPlot(rf.boston)
```

![](Lab_Decision_Trees_files/figure-markdown_github/unnamed-chunk-24-1.png)

The results indicate that across all of the trees considered in the random forest, the wealth level of the community (lstat) and the house size (rm) are by far the two most important variables.

Boosting
========

Here we use the gbm package, and within it the gbm() function, to fit boosted regression trees to the Boston data set. We run gbm() with the option distribution="gaussian" since this is a regression problem; if it were a binary classification problem, we would use distribution="bernoulli". The argument n.trees=5000 indicates that we want 5000 trees, and the option interaction.depth=4 limits the depth of each tree.

``` r
library(gbm)
```

    ## Loaded gbm 2.1.5

``` r
set.seed(1)
boost.boston = gbm(medv ~ ., data = Boston[train,],
                 distribution = "gaussian",
                 n.trees = 5000, 
                 interaction.depth = 4)
```

The summary() function produces a relative influence plot and also outputs the relative influence statistics.

``` r
summary(boost.boston)
```

![](Lab_Decision_Trees_files/figure-markdown_github/unnamed-chunk-26-1.png)

    ##             var    rel.inf
    ## lstat     lstat 37.0661275
    ## rm           rm 25.3533123
    ## dis         dis 11.7903016
    ## crim       crim  8.0388750
    ## black     black  4.2531659
    ## nox         nox  3.5058570
    ## age         age  3.4868724
    ## ptratio ptratio  2.2500385
    ## indus     indus  1.7725070
    ## tax         tax  1.1836592
    ## chas       chas  0.7441319
    ## rad         rad  0.4274311
    ## zn           zn  0.1277206

We see that lstat and rm are by far the most important variables. We can also produce partial dependence plots for these two variables. These plots illustrate the marginal effect of the selected variables on the response after integrating out the other variables. In this case, as we might expect, median house prices are increasing with rm and decreasing with lstat.

``` r
par(mfrow = c(1, 2)) 
plot(boost.boston, i = "rm") 
```

![](Lab_Decision_Trees_files/figure-markdown_github/unnamed-chunk-27-1.png)

``` r
plot(boost.boston, i = "lstat")
```

![](Lab_Decision_Trees_files/figure-markdown_github/unnamed-chunk-27-2.png)

We now use the boosted model to predict medv on the test set

``` r
yhat.boost = predict(boost.boston, newdata = Boston[-train,], 
                   n.trees = 5000)

mean((yhat.boost - boston.test)^2)
```

    ## [1] 10.81479

The test MSE obtained is 10.8; superior to the test MSE for random forests and to that for bagging. If we want to, we can perform boosting with a different value of the shrinkage parameter λ. The default value is 0.001, but this is easily modified. Here we take λ = 0.2.

``` r
boost.boston = gbm(medv ~ ., data = Boston[train,],
                 distribution = "gaussian",
                 n.trees = 5000, interaction.depth = 4,
                 shrinkage = 0.2, 
                 verbose = F)

yhat.boost = predict(boost.boston, newdata = Boston[-train,], 
                     n.trees = 5000)

mean((yhat.boost - boston.test)^2)
```

    ## [1] 11.51109

In this case, using λ = 0.2 leads to a higher test MSE than λ = 0.001.
