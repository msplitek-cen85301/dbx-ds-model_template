# Databricks notebook source
# MAGIC %md
# MAGIC # Model template

# COMMAND ----------

# MAGIC %md
# MAGIC ## Libraries

# COMMAND ----------

# MAGIC %md
# MAGIC ## Base load
# MAGIC    - target definiton?
# MAGIC    - undersampling

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature addition

# COMMAND ----------

# vzorek dat pro testovani a vyvoj

data = spark.sql('select * from sbx_ci_catalog.msp_database.cm2_baze_pokusny_vzorek')

data.display()data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preparation
# MAGIC    - remove columns with all missing values
# MAGIC    - missing imputation
# MAGIC    - standard scaling
# MAGIC    - one hot encoding

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature selection
# MAGIC    - univariate selection
# MAGIC    - selection with penalization of correlated predictors (GLO - Group Lasso with Overlap)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test/Valid split

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model definition
# MAGIC    - we want to do ensamble models, so we will define several different models here
# MAGIC    - logistic regression, XGBoost, Random forest

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter tuning
# MAGIC    - Grid search
# MAGIC    - Bayesian optimization

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model training
# MAGIC    - separate models
# MAGIC    - ensamble model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation
# MAGIC    - ROC, Gini
# MAGIC    - Lift, Gain chart
# MAGIC    - Confusion matrix, accuracy, F1 score, precision, and recall
# MAGIC    - Features with high intercorrelation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calibration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable explanation
# MAGIC    - feature importances
# MAGIC    - shapley
# MAGIC    - partial dependence plot

# COMMAND ----------

# MAGIC %md
# MAGIC ## Documentation
# MAGIC    - write Evaluation and Explanation into html file

# COMMAND ----------

# MAGIC %md
# MAGIC ## Choice of tempate complexity
# MAGIC    - parameters for each step

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save final model
