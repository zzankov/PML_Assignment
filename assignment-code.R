# load, randomize and subset data
rm(list=ls())
setwd("./Practical Machine Learning/PML-Assignment/")
library(caret)
library(AppliedPredictiveModeling)
library(doMC)
registerDoMC(cores=8)
data <- read.csv("pml-training.csv", stringsAsFactors = F)
set.seed(1986)
data <- data[sample(nrow(data),replace = F),]
testing <- read.csv("pml-testing.csv", stringsAsFactors = F)
inTrain <- createDataPartition(data$classe,p = 0.6, list = F)
# removing the first column. The outcome is ordered and thus strongly correlated with the number of the row
training <- data[inTrain, 2:dim(data)[2]]
cv <- data[-inTrain, 2:dim(data)[2]]
testing <- testing[,-1]
rm(data,inTrain)

# Set outcome to factor variable
training$classe <- as.factor(training$classe)
cv$classe <- as.factor(cv$classe)
training$user_name <- as.factor(training$user_name)
cv$user_name <- as.factor(cv$user_name)
testing$user_name <- as.factor(testing$user_name)

# remove near zero variance variables
nzVars <- nearZeroVar(training)
training <- training[,-nzVars]
cv <- cv[,-nzVars]
testing <- testing[,-nzVars]

# removing variables which have sparce data with a lot of NA values
apply(is.na(training),2,sum)
subsVect <- as.vector(apply(is.na(training),2,sum)==0)
training <- training[,subsVect]
cv <- cv[,subsVect]
testing <- testing[,subsVect]

# Nothing left to impute
sum(is.na(training))
sum(is.na(cv))
sum(is.na(testing))

# make data the appropriate format
for (i in 1:(dim(training)[2]-1)) {
  if (names(training)[i]=='cvtd_timestamp') {
    training$cvtd_timestamp <- as.Date(training$cvtd_timestamp)
    cv$cvtd_timestamp <- as.Date(cv$cvtd_timestamp)
    testing$cvtd_timestamp <- as.Date(testing$cvtd_timestamp)
  } else if (names(training)[i]!='user_name') {
    training[,i] <- as.numeric(training[,i])
    cv[,i] <- as.numeric(cv[,i])
    testing[,i] <- as.numeric(testing[,i])
  }
}

# Removing correlated variables
varCor <- cor(training[,-c(1,4,58)])
summary(varCor[upper.tri(varCor)])
highlyCorVar <- findCorrelation(varCor,cutoff = 0.9)
#training <- training[,-highlyCorVar]
#cv <- cv[,-highlyCorVar]
#testing <- cv[,-highlyCorVar]

# Exploring for linear combinations
comboInfo <- findLinearCombos(training[,-c(1,4,58)])
comboInfo

transparentTheme(trans=0.4)
featurePlot(x = training[,13:17],y = training$classe,
            plot='pairs',autokey = list(columns=5))

## Start fitting models on the whole dataset without trainControl
# Fit control to be used with certain models
fitControl <- trainControl(method = 'repeatedcv',
                           number = 10,
                           repeats = 10,
                           classProbs = TRUE)

# Logistic Regression
accur.tr <- numeric(5)
accur.cv <- numeric(5)

glm.fitA <- glm(I(classe=='A') ~ ., data = training, family = binomial)
glm.fitB <- glm(I(classe=='B') ~ ., data = training, family = binomial)
glm.fitC <- glm(I(classe=='C') ~ ., data = training, family = binomial)
glm.fitD <- glm(I(classe=='D') ~ ., data = training, family = binomial)
glm.fitE <- glm(I(classe=='E') ~ ., data = training, family = binomial)
prob.df.train <- data.frame(A = predict(glm.fitA,type = 'response'),
                            B = predict(glm.fitB,type = 'response'),
                            C = predict(glm.fitC,type = 'response'),
                            D = predict(glm.fitD,type = 'response'),
                            E = predict(glm.fitE,type = 'response'))
prob.df.cv <- data.frame(A = predict(glm.fitA,cv,type = 'response'),
                         B = predict(glm.fitB,cv,type = 'response'),
                         C = predict(glm.fitC,cv,type = 'response'),
                         D = predict(glm.fitD,cv,type = 'response'),
                         E = predict(glm.fitE,cv,type = 'response'))

pred.df.train <- as.factor(apply(prob.df.train,1,function(x) names(prob.df.train)[which(x==max(x))]))
train.accuracy <- mean(pred.df.train == training$classe)
train.accuracy
accur.tr[1] <- train.accuracy

pred.df.cv <- apply(prob.df.cv,1,function(x) names(prob.df.cv)[which(x==max(x))])
pred.df.cv[which(lapply(pred.df.cv,length)>1)] <- NA

cv.accuracy <- mean(pred.df.cv == cv$classe)
cv.accuracy
accur.cv[1] <- cv.accuracy

prob.df.test <- data.frame(A = predict(glm.fitA,testing,type = 'response'),
                         B = predict(glm.fitB,testing,type = 'response'),
                         C = predict(glm.fitC,testing,type = 'response'),
                         D = predict(glm.fitD,testing,type = 'response'),
                         E = predict(glm.fitE,testing,type = 'response'))

pred.df.test <- as.factor(apply(prob.df.test,1,function(x) names(prob.df.test)[which(x==max(x))]))
pred.df.test

# try out polynomials
a <- cbind(training[,-58],training[,-c(1,4,58)]^2,training[,58])
b <- cbind(cv[,-58],cv[,-c(1,4,58)]^2,cv[,58])
c <- cbind(testing[,-58],testing[,-c(1,4,58)]^2)

names(a) <- c(1:(ncol(a)-1),'classe')
names(b) <- c(1:(ncol(b)-1),'classe')
names(c) <- c(1:(ncol(c)))

glm.fitA <- glm(I(classe=='A') ~ ., data = a, family = binomial)
glm.fitB <- glm(I(classe=='B') ~ ., data = a, family = binomial)
glm.fitC <- glm(I(classe=='C') ~ ., data = a, family = binomial)
glm.fitD <- glm(I(classe=='D') ~ ., data = a, family = binomial)
glm.fitE <- glm(I(classe=='E') ~ ., data = a, family = binomial)
prob.df.train <- data.frame(A = predict(glm.fitA,type = 'response'),
                            B = predict(glm.fitB,type = 'response'),
                            C = predict(glm.fitC,type = 'response'),
                            D = predict(glm.fitD,type = 'response'),
                            E = predict(glm.fitE,type = 'response'))
prob.df.cv <- data.frame(A = predict(glm.fitA,b,type = 'response'),
                         B = predict(glm.fitB,b,type = 'response'),
                         C = predict(glm.fitC,b,type = 'response'),
                         D = predict(glm.fitD,b,type = 'response'),
                         E = predict(glm.fitE,b,type = 'response'))

pred.df.train <- as.factor(apply(prob.df.train,1,function(x) names(prob.df.train)[which(x==max(x))]))
train.accuracy <- mean(pred.df.train == training$classe)
train.accuracy 
accur.tr[2] <- train.accuracy

pred.df.cv <- apply(prob.df.cv,1,function(x) names(prob.df.cv)[which(x==max(x))])
pred.df.cv[which(lapply(pred.df.cv,length)>1)] <- NA

cv.accuracy <- mean(pred.df.cv == cv$classe)
cv.accuracy
accur.cv[2] <- cv.accuracy

prob.df.test <- data.frame(A = predict(glm.fitA,c,type = 'response'),
                           B = predict(glm.fitB,c,type = 'response'),
                           C = predict(glm.fitC,c,type = 'response'),
                           D = predict(glm.fitD,c,type = 'response'),
                           E = predict(glm.fitE,c,type = 'response'))

pred.df.test <- rbind(pred.df.test,as.factor(apply(prob.df.test,1,function(x) names(prob.df.test)[which(x==max(x))])))

a <- cbind(training[,-58],training[,-c(1,4,58)]^2,training[,-c(1,4,58)]^3,training[,58])
b <- cbind(cv[,-58],cv[,-c(1,4,58)]^2,cv[,-c(1,4,58)]^3,cv[,58])
c <- cbind(testing[,-58],testing[,-c(1,4,58)]^2,testing[,-c(1,4,58)]^3)

names(a) <- c(1:(ncol(a)-1),'classe')
names(b) <- c(1:(ncol(b)-1),'classe')
names(c) <- c(1:(ncol(c)))

glm.fitA <- glm(I(classe=='A') ~ ., data = a, family = binomial)
glm.fitB <- glm(I(classe=='B') ~ ., data = a, family = binomial)
glm.fitC <- glm(I(classe=='C') ~ ., data = a, family = binomial)
glm.fitD <- glm(I(classe=='D') ~ ., data = a, family = binomial)
glm.fitE <- glm(I(classe=='E') ~ ., data = a, family = binomial)
prob.df.train <- data.frame(A = predict(glm.fitA,type = 'response'),
                            B = predict(glm.fitB,type = 'response'),
                            C = predict(glm.fitC,type = 'response'),
                            D = predict(glm.fitD,type = 'response'),
                            E = predict(glm.fitE,type = 'response'))
prob.df.cv <- data.frame(A = predict(glm.fitA,b,type = 'response'),
                         B = predict(glm.fitB,b,type = 'response'),
                         C = predict(glm.fitC,b,type = 'response'),
                         D = predict(glm.fitD,b,type = 'response'),
                         E = predict(glm.fitE,b,type = 'response'))

pred.df.train <- apply(prob.df.train,1,function(x) names(prob.df.train)[which(x==max(x))])
pred.df.train[which(lapply(pred.df.train,length)>1)] <- NA
# pred.df.train <- as.factor(apply(prob.df.train,1,function(x) names(prob.df.train)[which(x==max(x))]))
train.accuracy <- mean(pred.df.train == training$classe)
train.accuracy 
accur.tr[3] <- train.accuracy

pred.df.cv <- apply(prob.df.cv,1,function(x) names(prob.df.cv)[which(x==max(x))])
pred.df.cv[which(lapply(pred.df.cv,length)>1)] <- NA

cv.accuracy <- mean(pred.df.cv == cv$classe)
cv.accuracy
accur.cv[3] <- cv.accuracy

prob.df.test <- data.frame(A = predict(glm.fitA,c,type = 'response'),
                           B = predict(glm.fitB,c,type = 'response'),
                           C = predict(glm.fitC,c,type = 'response'),
                           D = predict(glm.fitD,c,type = 'response'),
                           E = predict(glm.fitE,c,type = 'response'))

pred.df.test <- rbind(pred.df.test,as.factor(apply(prob.df.test,1,function(x) names(prob.df.test)[which(x==max(x))])))

a <- cbind(training[,-58],training[,-c(1,4,58)]^2,training[,-c(1,4,58)]^3,training[,-c(1,4,58)]^4,training[,58])
b <- cbind(cv[,-58],cv[,-c(1,4,58)]^2,cv[,-c(1,4,58)]^3,cv[,-c(1,4,58)]^4,cv[,58])
c <- cbind(testing[,-58],testing[,-c(1,4,58)]^2,testing[,-c(1,4,58)]^3,testing[,-c(1,4,58)]^4)

names(a) <- c(1:(ncol(a)-1),'classe')
names(b) <- c(1:(ncol(b)-1),'classe')
names(c) <- c(1:(ncol(c)))
              
glm.fitA <- glm(I(classe=='A') ~ ., data = a, family = binomial)
glm.fitB <- glm(I(classe=='B') ~ ., data = a, family = binomial)
glm.fitC <- glm(I(classe=='C') ~ ., data = a, family = binomial)
glm.fitD <- glm(I(classe=='D') ~ ., data = a, family = binomial)
glm.fitE <- glm(I(classe=='E') ~ ., data = a, family = binomial)
prob.df.train <- data.frame(A = predict(glm.fitA,type = 'response'),
                            B = predict(glm.fitB,type = 'response'),
                            C = predict(glm.fitC,type = 'response'),
                            D = predict(glm.fitD,type = 'response'),
                            E = predict(glm.fitE,type = 'response'))
prob.df.cv <- data.frame(A = predict(glm.fitA,b,type = 'response'),
                         B = predict(glm.fitB,b,type = 'response'),
                         C = predict(glm.fitC,b,type = 'response'),
                         D = predict(glm.fitD,b,type = 'response'),
                         E = predict(glm.fitE,b,type = 'response'))

pred.df.train <- apply(prob.df.train,1,function(x) names(prob.df.train)[which(x==max(x))])
pred.df.train[which(lapply(pred.df.train,length)>1)] <- pred.df.train[which(lapply(pred.df.train,length)>1)][[1]][1]
train.accuracy <- mean(pred.df.train == training$classe)
train.accuracy 
accur.tr[4] <- train.accuracy

pred.df.cv <- apply(prob.df.cv,1,function(x) names(prob.df.cv)[which(x==max(x))])
pred.df.cv[which(lapply(pred.df.cv,length)>1)] <- pred.df.cv[which(lapply(pred.df.cv,length)>1)][[1]][1]

cv.accuracy <- mean(pred.df.cv == cv$classe)
cv.accuracy
accur.cv[4] <- cv.accuracy

prob.df.test <- data.frame(A = predict(glm.fitA,c,type = 'response'),
                           B = predict(glm.fitB,c,type = 'response'),
                           C = predict(glm.fitC,c,type = 'response'),
                           D = predict(glm.fitD,c,type = 'response'),
                           E = predict(glm.fitE,c,type = 'response'))

temp <- apply(prob.df.test,1,function(x) names(prob.df.test)[which(x==max(x))])
temp[which(lapply(temp,length)>1)] <- temp[which(lapply(temp,length)>1)][[1]][1]
pred.df.test <- rbind(pred.df.test, as.factor(unlist(temp)))

a <- cbind(training[,-58],training[,-c(1,4,58)]^2,training[,-c(1,4,58)]^3,training[,-c(1,4,58)]^4,training[,-c(1,4,58)]^5,training[,58])
b <- cbind(cv[,-58],cv[,-c(1,4,58)]^2,cv[,-c(1,4,58)]^3,cv[,-c(1,4,58)]^4,cv[,-c(1,4,58)]^5,cv[,58])
c <- cbind(testing[,-58],testing[,-c(1,4,58)]^2,testing[,-c(1,4,58)]^3,testing[,-c(1,4,58)]^4,testing[,-c(1,4,58)]^5)

names(a) <- c(1:(ncol(a)-1),'classe')
names(b) <- c(1:(ncol(b)-1),'classe')
names(c) <- c(1:(ncol(c)))
              
glm.fitA <- glm(I(classe=='A') ~ ., data = a, family = binomial)
glm.fitB <- glm(I(classe=='B') ~ ., data = a, family = binomial)
glm.fitC <- glm(I(classe=='C') ~ ., data = a, family = binomial)
glm.fitD <- glm(I(classe=='D') ~ ., data = a, family = binomial)
glm.fitE <- glm(I(classe=='E') ~ ., data = a, family = binomial)
prob.df.train <- data.frame(A = predict(glm.fitA,type = 'response'),
                            B = predict(glm.fitB,type = 'response'),
                            C = predict(glm.fitC,type = 'response'),
                            D = predict(glm.fitD,type = 'response'),
                            E = predict(glm.fitE,type = 'response'))
prob.df.cv <- data.frame(A = predict(glm.fitA,b,type = 'response'),
                         B = predict(glm.fitB,b,type = 'response'),
                         C = predict(glm.fitC,b,type = 'response'),
                         D = predict(glm.fitD,b,type = 'response'),
                         E = predict(glm.fitE,b,type = 'response'))

pred.df.train <- as.factor(apply(prob.df.train,1,function(x) names(prob.df.train)[which(x==max(x))]))
train.accuracy <- mean(pred.df.train == training$classe)
train.accuracy 
accur.tr[5] <- train.accuracy

pred.df.cv <- apply(prob.df.cv,1,function(x) names(prob.df.cv)[which(x==max(x))])
pred.df.cv[which(lapply(pred.df.cv,length)>1)] <- pred.df.cv[which(lapply(pred.df.cv,length)>1)][[1]][1]

cv.accuracy <- mean(pred.df.cv == cv$classe)
cv.accuracy
accur.cv[5] <- cv.accuracy

prob.df.test <- data.frame(A = predict(glm.fitA,c,type = 'response'),
                           B = predict(glm.fitB,c,type = 'response'),
                           C = predict(glm.fitC,c,type = 'response'),
                           D = predict(glm.fitD,c,type = 'response'),
                           E = predict(glm.fitE,c,type = 'response'))

temp <- apply(prob.df.test,1,function(x) names(prob.df.test)[which(x==max(x))])
temp[which(lapply(temp,length)>1)] <- temp[which(lapply(temp,length)>1)][[1]][1]
pred.df.test <- rbind(pred.df.test, as.factor(unlist(temp)))

par('bg' = 'white')
plot(1:5,accur.tr,type = 'l', col = 'blue', lwd = 2, frame.plot=F,
     ylim = c(min(accur.tr,accur.cv),max(accur.tr,accur.cv)),
     xlab = 'Degree of polynomial',
     ylab = 'Accuracy', main = 'Logistic Regression Fit')
lines(accur.cv, col = 'red', lwd = 2)
grid()

# Boosted Logistic Regression - a problem with Logitboost is that it returns NA for some predictions
# At the NA instances we do not really know which one is the correct class
# we fit multiple models until we are able to give a prediction for all 20 test examples
LogitBoostFit <- train(classe ~ ., data = training,
                       method = 'LogitBoost')

LogitBoostFit2 <- train(classe ~ ., data = training,
                       method = 'LogitBoost',
                       tuneGrid = expand.grid(nIter = seq(10,100,by = 10)))

LogitBoostFit3 <- train(classe ~ ., data = training,
                        method = 'LogitBoost',
                        tuneGrid = expand.grid(nIter = 100))

LogitBoostFit4 <- train(classe ~ ., data = training,
                        method = 'LogitBoost',
                        tuneGrid = expand.grid(nIter = 100),
                        preProcess = c('scale','center'))
LogitBoostFit4
confusionMatrix(predict(LogitBoostFit,cv),cv$classe)
confusionMatrix(predict(LogitBoostFit2,cv),cv$classe)
confusionMatrix(predict(LogitBoostFit3,cv),cv$classe)
confusionMatrix(predict(LogitBoostFit4,cv),cv$classe)

predict(LogitBoostFit,testing)
predict(LogitBoostFit2,testing)
predict(LogitBoostFit3,testing)
predict(LogitBoostFit4,testing)
# LDA
# not the best accuracy 73% ~ 75%. Not surprising since the data doesn't really seem linear

lda.fit <- train(classe ~ ., data = training,
                 method = 'lda')
confusionMatrix(predict(lda.fit,cv),cv$classe)
lda.fit.prep <- train(classe ~ ., data = training,
                     method = 'lda',
                     preProcess = c('center','scale'))
confusionMatrix(predict(lda.fit,cv),)
lda.fit2 <- train(classe ~ ., data = training,
                 method = 'lda2',
                 tuneGrid = expand.grid(dimen = 1:100))

# QDA - very good accuracy
qda.fit <- train(classe ~ ., data = training[,-highlyCorVar],
                 method = 'qda')
qda.fit.prep <- train(classe ~ ., data = training[,-highlyCorVar],
                      method = 'qda',
                      preProcess = c('center','scale'))
qda.fit2 <- train(classe ~ ., data = training[,-highlyCorVar],
                 method = 'qda',
                 trControl = fitControl)
qda.fit2.prep <- train(classe ~ ., data = training[,-highlyCorVar],
                  method = 'qda',
                  trControl = fitControl,
                  preProcess = c('center','scale'))

confusionMatrix(predict(qda.fit,cv),cv$classe)
confusionMatrix(predict(qda.fit.prep,cv),cv$classe)
confusionMatrix(predict(qda.fit2,cv),cv$classe)
confusionMatrix(predict(qda.fit2.prep,cv),cv$classe)
predict(qda.fit,testing)
predict(qda.fit2,testing)
predict(qda.fit.prep,testing)
predict(qda.fit2.prep,testing)

predict(LogitBoostFit3, testing)
predict(LogitBoostFit4, testing)

# KNN fit - not expected to perform well because of the curse of dimentionality
# turns out knn performs great especially knn.fit2
knn.fit <- train(classe ~ ., data = training,
                 method = 'knn',
                 tuneGrid = expand.grid(k=5),
                 preProcess = c('center','scale'))
confusionMatrix(predict(knn.fit,cv),cv$classe)
predict(knn.fit,testing)

knn.fit2 <- train(classe ~ ., data = training,
                  method = 'kknn',
                  preProcess = c('center','scale'))
confusionMatrix(predict(knn.fit2,cv),cv$classe)
predict(knn.fit2,testing)

# A few other Discriminant Analysis models just to compare performance
# flexible discriminant analysis
fda.fit <- train(classe ~ ., data = training,
                 method = 'fda',
                 preProcess = c('center','scale'))

fda.fit2 <- train(classe ~ ., data = training,
                 method = 'fda',
                 preProcess = c('center','scale'),
                 tuneGrid = expand.grid(degree = 2,nprune = c(100,150))

# Time to fit the model

# A couple of simpler tree based models
# These do not perform very well. between 65% and 60% accuracy
# ctree crashes the system

# set.seed(54321)
# ctreeMod <- train(classe ~ ., data = training,
#                   method = 'ctree')
# ctreeMod <- train(classe ~ ., data = training,
#                   method = 'ctree',
#                   trControl = fitControl)
# 
# library(rattle)
# fancyRpartPlot(ctreeMod$finalModel)

## RPART has poor performance
set.seed(54321)
rpart.fit <- train(classe ~ ., data = training,
                  method = 'rpart',tuneGrid = expand.grid(cp=seq(0,1,by=0.1)))
rpart.fit2 <- train(classe ~ ., data = training,
                  method = 'rpart',
                  trControl = fitControl)
rpart.fit
rpart.fit2
confusionMatrix(predict(rpart.fit,cv),cv$classe)
confusionMatrix(predict(rpart.fit2,cv),cv$classe)

# to make it work highly correlated variables have to be excluded
# maybe also try lda, fda, mda, hda, hdda, LogitBoost, nb

# Trying out a random forest model -- high accuracy! 94%
set.seed(54321)
rf.fit <- train(classe ~ ., data = training,
               method = 'rf')
confusionMatrix(predict(rf.fit,training),training$classe)
confusionMatrix(predict(rf.fit,cv),cv$classe)

# preProcessing the data makes no difference
# it takes quite a while to train this model

# rf.fit2 <- train(classe ~ ., data = training,
#                  method = 'rf',
#                  preProcess = c('center','scale'))
# confusionMatrix(predict(rf.fit2,training),training$classe)
# confusionMatrix(predict(rf.fit2,cv),cv$classe)
# 
# rf.fit3 <- train(classe ~ ., data = training,
#                 method = 'rf',
#                 preProcess = c('center','scale')
#                 trControl = fitControl)

# Bagging with trees is also a very accurate model 95%
# prohibitively computationally expensive to fit
set.seed(54321)
treebagMod <- train(classe ~ ., data = training, 
                    method = 'treebag')

# Boosting with trees model This mod
# prohibitively expensive to fit
# tuning  the training parameters
gbmGrid <-  expand.grid(interaction.depth = 5,
                        n.trees = 900,
                        shrinkage = 0.1)
set.seed(54321)
gbm.fit <- train(classe ~ ., data = training,
                 preProcess = c('center','scale'),
                 method = 'gbm',
                 verbose = FALSE,
                 tuneGrid = gbmGrid)

# Fitting an SVM
# computationally infeasable
svmPoly.fit <- train(classe ~ ., data = training,
                     method = 'svmPoly',
                     preProcess = c('center','scale')
                     #,tuneGrid = expand.grid(degree = 1:5)
                     )
svmRadial.Fit <- train(classe ~ .,data = training,
                       method = 'svmRadial',
                       preProcess = c('center','scale'))

### Create files for submission
final.predictions <- predict(rf.fit,testing)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0(getwd(),"/predictions/","problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(final.predictions)
