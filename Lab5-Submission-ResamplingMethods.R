
  
#STEP 1: Loading the dataset
  PimaIndiansDiabetes <-
    readr::read_csv(
      "data/pima-indians-diabetes.csv")

  #NAIVE BAYES
#STEP 2: Splitting the dataset ----
  train_index <- createDataPartition(PimaIndiansDiabetes$`No. of pregnancies`, # nolint
                                     p = 0.80, list = FALSE)
  PimaIndiansDiabetes_train <- PimaIndiansDiabetes[train_index, ]
  PimaIndiansDiabetes_test <- PimaIndiansDiabetes[-train_index, ]
  
  
  
#STEP 3a: using "NaiveBayes()" function in the "e1071" package ----
  PimaIndiansDiabetes_model_nb_e1071 <- # nolint
    e1071::naiveBayes(`No. of pregnancies` ~ .,
                     data = PimaIndiansDiabetes_train)
  
  
#STEP 3b: using "NaiveBayes()" function in the "klaR" package 
  PimaIndiansDiabetes_model_nb_klaR <-
    klaR::NaiveBayes(`No. of pregnancies` ~ .,
                     data = PimaIndiansDiabetes_train)
  
#STEP 4:Testing trained Naive Bayes model
  predictions_nb_e1071 <-
    predict(PimaIndiansDiabetes_model_nb_e1071,
            PimaIndiansDiabetes_test[, 1:9])

#STEP 5:Printing results
  print(PimaIndiansDiabetes_model_nb_e1071)
  caret::confusionMatrix(predictions_nb_e1071,
                         PimaIndiansDiabetes_test$`No. of pregnancies`)

#STEP 6:The confusion matrix
  plot(table(predictions_nb_e1071,
             PimaIndiansDiabetes_test$`No. of pregnancies`))
  
  predictions_nb_e1071 <-
    predict(defaulter_dataset_model_nb_e1071,
            defaulter_dataset_test[, 1:25])
  
  

  
#STEP 1: Loading the dataset
  PimsIndiansDiabetes <-
    readr::read_csv(
      "data/pima-indians-diabetes.csv")
  
#STEP 2: Splitting the dataset into training and testing datasets
  train_index <- createDataPartition(PimaIndiansDiabetes$`No. of pregnancies`, p = 0.65, list = FALSE)
  PimaIndiansDiabetes_train <- PimaIndiansDiabetes[train_index, ]
  PimaIndiansDiabetes_test <- PimaIndiansDiabetes[-train_index, ]
  
  
#STEP 3:Training a logistic regression model
 train_control <- trainControl(method = "boot", number = 500)
  
 PimaIndiansDiabetes_model_lm <- 
   caret::train(`No. of pregnancies` ~ 
                 Glucose + `Blood Pressure` + `Skin Thickness`
                + Insulin + `Diabetes predigree function`+
                  + Age + Class,
   data = PimaIndiansDiabetes_train,
   trControl = train_control,
   na.action = na.omit,
   method = "lm",
   metric = "RMSE"
 )
 
 
#STEP 4: Testing the model
 predictions_lm <- predict(PimaIndiansDiabetes_model_lm,
                           PimaIndiansDiabetes_test[, 1:9])
 

 #STEP 5: Viewing the accuracy and the predicted values
 print(PimaIndiansDiabetes_model_lm)
 print(predictions_lm)
 
 new_data <-
   data.frame( `No. of pregnancies` = c(4), 
               Glucose = c(160),
               `Blood Pressure` = c(149),
               triceps = c(50),
               insulin = c(450),
               mass = c(50),
               pedigree = c(1.9),
               age = c(48), check.names = FALSE)
 

#STEP 6: Using the model to predict the output based on unseen data
 predictions_lm_new_data <- 
   predict(PimaIndiansDiabetes_model_lm, new_data)
   predict(PimaIndiansDiabetes_model_lm, new_data)
   
#STEP 7: The output below
   print(predictions_lm_new_data)
   
   #---cross validation----
#STEP 1: Loading the dataset   
   data("PimaIndiansDiabetes")
   
#STEP 2: Splitting the dataset into training and testing datasets
   train_index <- createDataPartition(PimaIndiansDiabetes$`No. of pregnancies`,
                                      p = 0.60, list = FALSE)
   PimaIndiansDiabetes_train <- PimaIndiansDiabetes[train_index, ]
   PimaIndiansDiabetes_test <- PimaIndiansDiabetes[-train_index, ]
   
   #--REGRESSION--
   #STEP 3a: using 10-fold cross validation
   train_control <- trainControl(method = "cv", number = 10)
   
   PimaIndiansDiabetes_model_lm <-
     caret::train(diabetes ~ .,
                  data = PimaIndiansDiabetes_train,
                  trControl = train_control, na.action = na.omit,
                  method = "glm", metric = "Accuracy")
   
   
   #STEP 3b: Testing the trained linear model
   predictions_lm <- predict(PimaIndiansDiabetes_model_lm, PimaIndiansDiabetes_test[, -9])
   
   #STEP 4: View the Accuracy and the predicted values
   print(PimaIndiansDiabetes_model_lm)
   print(predictions_lm)
   
   
   #CLASSIFICATION
   
   #STEP 5: LDA classifier based on a 5-fold cross validation
   train_control <- trainControl(method = "cv", number = 5)
   
   PimaIndiansDiabetes_model_lda <-
     caret::train(`diabetes` ~ ., data = PimaIndiansDiabetes_train,
                  trControl = train_control, na.action = na.omit, method = "lda2",
                  metric = "Accuracy")
   #STEP 6: Testing the trained LDA model
   predictions_lda <- predict(PimaIndiansDiabetes_model_lda,
                              PimaIndiansDiabetes_test[, 1:9])
   
   #STEP 7: Viewing the summary of the model and view the confusion matrix
   
   print(PimaIndiansDiabetes_model_lda)
   caret::confusionMatrix(predictions_lda, PimaIndiansDiabetes_test$diabetes)
   
   
   PimaIndiansDiabetes_model_nb <-
     e1071::naiveBayes(diabetes ~ ., data = PimaIndiansDiabetes_train)
   
   
   predictions_nb_e1071 <-
     predict(PimaIndiansDiabetes_model_nb, PimaIndiansDiabetes_test[, 1:9])
   
   #STEP 7: View a summary of the naive Bayes model and the confusion matrix 
   print(PimaIndiansDiabetes_model_nb)
   caret::confusionMatrix(predictions_nb_e1071, PimaIndiansDiabetes_test$diabetes)
   
   #STEP 8: Classification: SVM with Repeated k-fold Cross Validation----
   train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
   
   PimaIndiansDiabetes_model_svm <-
     caret::train(diabetes ~ ., data = PimaIndiansDiabetes_train,
                  trControl = train_control, na.action = na.omit,
                  method = "svmLinearWeights2", metric = "Accuracy")
   
   
   predictions_svm <- predict(PimaIndiansDiabetes_model_svm, PimaIndiansDiabetes_test[, 1:9])
   
   
   print(PimaIndiansDiabetes_model_svm)
   caret::confusionMatrix(predictions_svm, PimaIndiansDiabetes_test$diabetes)
   
   ##STEP 9: Classification: Naive Bayes with Leave One Out Cross Validation ----
   train_control <- trainControl(method = "LOOCV")
   
   PimaIndiansDiabetes_model_nb_loocv <-
     caret::train(diabetes ~ ., data = PimaIndiansDiabetes_train,
                  trControl = train_control, na.action = na.omit,
                  method = "naive_bayes", metric = "Accuracy")
   
   
   predictions_nb_loocv <-
     predict(PimaIndiansDiabetes_model_nb_loocv, PimaIndiansDiabetes_test[, 1:9])
   
   
   print(PimaIndiansDiabetes_model_nb_loocv)
   caret::confusionMatrix(predictions_nb_loocv, PimaIndiansDiabetes_test$diabetes)
   