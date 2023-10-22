Business Intelligence Project
================
<Specify your name here>
<Specify the date when you submitted the lab>

- [Business Intelligence Lab Submission
  Markdown](#business-intelligence-lab-submission-markdown)
- [Student Details](#student-details)
- [Setup Chunk](#setup-chunk)
- [Step 1: Install and Load Packages](#step-1-install-and-load-packages)
- [Step 2: Load Dataset](#step-2-load-dataset)
- [Step 3: Using Naives Bayes](#step-3-using-naives-bayes)
  - [Splitting the dataset](#splitting-the-dataset)
  - [Testing the trained Naive Bayes
    model](#testing-the-trained-naive-bayes-model)
  - [Printing Results](#printing-results)
  - [Confusion Matrix](#confusion-matrix)
- [Step 4: Using Bootstrapping](#step-4-using-bootstrapping)
  - [Load the dataset](#load-the-dataset)
  - [Split and train the dataset](#split-and-train-the-dataset)
  - [Training a logistic regression
    model](#training-a-logistic-regression-model)
  - [Test the Model](#test-the-model)
  - [View accuracy and the predicted
    valies](#view-accuracy-and-the-predicted-valies)
  - [Predict output on unseen data and print
    output](#predict-output-on-unseen-data-and-print-output)
- [Step 5: Using Cross Validation](#step-5-using-cross-validation)
  - [Load the dataset](#load-the-dataset-1)
  - [Split and train the dataset](#split-and-train-the-dataset-1)
  - [Training using 10-fold cross validation and testing the trained
    linear
    model](#training-using-10-fold-cross-validation-and-testing-the-trained-linear-model)
  - [View accuracy and the predicted
    valies](#view-accuracy-and-the-predicted-valies-1)
  - [LDA Classifier based on 5-fold cross
    validation](#lda-classifier-based-on-5-fold-cross-validation)
  - [Testing the trained LDA Model](#testing-the-trained-lda-model)
  - [Summary of the model](#summary-of-the-model)
  - [Confusion Matrix](#confusion-matrix-1)
  - [Classification: SVM with Repeated k-fold Cross
    Validation](#classification-svm-with-repeated-k-fold-cross-validation)
  - [Classification: Naive Bayes with Leave One Out Cross
    Validation](#classification-naive-bayes-with-leave-one-out-cross-validation)

# Business Intelligence Lab Submission Markdown

# Student Details

<table style="width:99%;">
<colgroup>
<col style="width: 43%" />
<col style="width: 38%" />
<col style="width: 17%" />
</colgroup>
<tbody>
<tr class="odd">
<td><strong>Student ID Numbers and Names of Group Members</strong></td>
<td><div class="line-block">1. 134982 - A - Austin Waswa</div>
<div class="line-block">2. 100230 - A - Richard Maana</div>
<div class="line-block">3. 134564 - A - Cynthia Omusundi</div></td>
<td></td>
</tr>
<tr class="even">
<td></td>
<td><strong>GitHub Classroom Group Name</strong></td>
<td>Luminosity</td>
</tr>
<tr class="odd">
<td><strong>Course Code</strong></td>
<td>BBT4206</td>
<td></td>
</tr>
<tr class="even">
<td><strong>Course Name</strong></td>
<td>Business Intelligence II</td>
<td></td>
</tr>
<tr class="odd">
<td><strong>Program</strong></td>
<td>Bachelor of Business Information Technology</td>
<td></td>
</tr>
<tr class="even">
<td><strong>Semester Duration</strong></td>
<td>21<sup>st</sup> August 2023 to 28<sup>th</sup> November 2023</td>
<td></td>
</tr>
</tbody>
</table>

# Setup Chunk

**Note:** the following “*KnitR*” options have been set as the defaults
in this markdown:  
`knitr::opts_chunk$set(echo = TRUE, warning = FALSE, eval = TRUE, collapse = FALSE, tidy.opts = list(width.cutoff = 80), tidy = TRUE)`.

More KnitR options are documented here
<https://bookdown.org/yihui/rmarkdown-cookbook/chunk-options.html> and
here <https://yihui.org/knitr/options/>.

``` r
knitr::opts_chunk$set(
    eval = TRUE,
    echo = TRUE,
    warning = FALSE,
    collapse = FALSE,
    tidy = TRUE
)
```

------------------------------------------------------------------------

**Note:** the following “*R Markdown*” options have been set as the
defaults in this markdown:

> output:
>
> github_document:  
> toc: yes  
> toc_depth: 4  
> fig_width: 6  
> fig_height: 4  
> df_print: default
>
> editor_options:  
> chunk_output_type: console

# Step 1: Install and Load Packages

We start by installing all the required packages

``` r
# STEP 1. Install and Load the Required Packages ----

#---- Installing and Loading the Required Packages ----
## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## klaR ----
if (require("klaR")) {
  require("klaR")
} else {
  install.packages("klaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## readr ----
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## LiblineaR ----
if (require("LiblineaR")) {
  require("LiblineaR")
} else {
  install.packages("LiblineaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naivebayes ----
if (require("naivebayes")) {
  require("naivebayes")
} else 
  {
  install.packages("naivebayes", dependencies = TRUE,
                   
                repos = "https://cloud.r-project.org")
  }
##mlbench
  if (!is.element("mlbench", installed.packages()[, 1])) {
    install.packages("mlbench", dependencies = TRUE)
  }
  require("mlbench")
  
```

# Step 2: Load Dataset

``` r
  PimaIndiansDiabetes <-
    readr::read_csv(
      "data/pima-indians-diabetes.csv")
```

# Step 3: Using Naives Bayes

### Splitting the dataset

``` r
  train_index <- createDataPartition(PimaIndiansDiabetes$Insulin, # nolint
                                     p = 0.80, list = FALSE)
  PimaIndiansDiabetes_train <- PimaIndiansDiabetes[train_index, ]
  PimaIndiansDiabetes_test <- PimaIndiansDiabetes[-train_index, ]
  
  
#using "NaiveBayes()" function in the "e1071" package ----
  PimaIndiansDiabetes_model_nb_e1071 <- # nolint
    e1071::NaiveBayes(diabetes ~ .,
                     data = PimaIndiansDiabetes_train)
  
#using "NaiveBayes()" function in the "klaR" package 
  PimaIndiansDiabetes_model_nb_klaR <-
    klaR::NaiveBayes(diabetes ~ .,
                     data = PimaIndiansDiabetes_train)
```

### Testing the trained Naive Bayes model

``` r
predictions_nb_e1071 <-
    predict(PimaIndiansDiabetes_model_nb_e1071,
            PimaIndiansDiabetes_test[, 1:9])
```

### Printing Results

``` r
  print(PimaIndiansDiabetes_model_nb_e1071)
  caret::confusionMatrix(predictions_nb_e1071,
                         PimaIndiansDiabetes_test$`No. of pregnancies`)
```

### Confusion Matrix

``` r
  plot(table(predictions_nb_e1071,
             PimaIndiansDiabetes_test$`No. of pregnancies`))
```

# Step 4: Using Bootstrapping

### Load the dataset

``` r
  PimsIndiansDiabetes <-
    readr::read_csv(
      "data/pima-indians-diabetes.csv")
```

### Split and train the dataset

``` r
#STEP 2: Splitting the dataset into training and testing datasets
  train_index <- createDataPartition(PimaIndiansDiabetes$`No. of pregnancies`, p = 0.65, list = FALSE)
  PimaIndiansDiabetes_train <- PimaIndiansDiabetes[train_index, ]
  PimaIndiansDiabetes_test <- PimaIndiansDiabetes[-train_index, ]
  
```

### Training a logistic regression model

``` r
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
 
```

### Test the Model

``` r
#STEP 4: Testing the model
 predictions_lm <- predict(PimaIndiansDiabetes_model_lm,
                           PimaIndiansDiabetes_test[, 1:9])
```

### View accuracy and the predicted valies

``` r
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
```

### Predict output on unseen data and print output

``` r
#STEP 6: Using the model to predict the output based on unseen data
 predictions_lm_new_data <- 
   predict(PimaIndiansDiabetes_model_lm, new_data)
   predict(PimaIndiansDiabetes_model_lm, new_data)
   
#STEP 7: The output below
   print(predictions_lm_new_data)
```

# Step 5: Using Cross Validation

### Load the dataset

``` r
  PimsIndiansDiabetes <-
    readr::read_csv(
      "data/pima-indians-diabetes.csv")
```

### Split and train the dataset

``` r
#STEP 2: Splitting the dataset into training and testing datasets
  train_index <- createDataPartition(PimaIndiansDiabetes$`No. of pregnancies`, p = 0.65, list = FALSE)
  PimaIndiansDiabetes_train <- PimaIndiansDiabetes[train_index, ]
  PimaIndiansDiabetes_test <- PimaIndiansDiabetes[-train_index, ]
  
```

### Training using 10-fold cross validation and testing the trained linear model

``` r
 
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
   
```

### View accuracy and the predicted valies

``` r
   #STEP 4: View the Accuracy and the predicted values
   print(PimaIndiansDiabetes_model_lm)
   print(predictions_lm)
   
```

### LDA Classifier based on 5-fold cross validation

``` r
#--CLASSIFICATION----
   #STEP 5: LDA classifier based on a 5-fold cross validation
   train_control <- trainControl(method = "cv", number = 5)
   
   PimaIndiansDiabetes_model_lda <-
     caret::train(`diabetes` ~ ., data = PimaIndiansDiabetes_train,
                  trControl = train_control, na.action = na.omit, method = "lda2",
                  metric = "Accuracy")
```

### Testing the trained LDA Model

``` r
   #STEP 6: Testing the trained LDA model
   predictions_lda <- predict(PimaIndiansDiabetes_model_lda,
                              PimaIndiansDiabetes_test[, 1:9])
```

### Summary of the model

``` r
   #STEP 7: Viewing the summary of the model and view the confusion matrix
   
   print(PimaIndiansDiabetes_model_lda)
   caret::confusionMatrix(predictions_lda, PimaIndiansDiabetes_test$diabetes)
   
   
   PimaIndiansDiabetes_model_nb <-
     e1071::naiveBayes(diabetes ~ ., data = PimaIndiansDiabetes_train)
   
   
   predictions_nb_e1071 <-
     predict(PimaIndiansDiabetes_model_nb, PimaIndiansDiabetes_test[, 1:9])
   
```

### Confusion Matrix

``` r
  print(PimaIndiansDiabetes_model_nb)
   caret::confusionMatrix(predictions_nb_e1071, PimaIndiansDiabetes_test$diabetes)
   
```

### Classification: SVM with Repeated k-fold Cross Validation

``` r
#STEP 8: Classification: SVM with Repeated k-fold Cross Validation----
   train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
   
   PimaIndiansDiabetes_model_svm <-
     caret::train(diabetes ~ ., data = PimaIndiansDiabetes_train,
                  trControl = train_control, na.action = na.omit,
                  method = "svmLinearWeights2", metric = "Accuracy")
   
   
   predictions_svm <- predict(PimaIndiansDiabetes_model_svm, PimaIndiansDiabetes_test[, 1:9])
   
   
   print(PimaIndiansDiabetes_model_svm)
   caret::confusionMatrix(predictions_svm, PimaIndiansDiabetes_test$diabetes)
   
```

### Classification: Naive Bayes with Leave One Out Cross Validation

``` r
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
   
   
```
