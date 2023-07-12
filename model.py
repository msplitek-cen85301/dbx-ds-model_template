# Databricks notebook source
# MAGIC %md
# MAGIC # Model template

# COMMAND ----------

# MAGIC %md
# MAGIC ## Libraries

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Base load
# MAGIC    - undersampling

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature addition

# COMMAND ----------

# vzorek dat pro testovani a vyvoj

data = spark.sql('select * from sbx_ci_catalog.msp_database.cm2_baze_pokusny_vzorek')

data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preparation
# MAGIC    - remove columns with all missing values
# MAGIC    - missing imputation
# MAGIC    - standard scaling
# MAGIC    - one hot encoding

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature selection
# MAGIC    - univariate selection
# MAGIC    - selection with penalization of correlated predictors (GLO - Group Lasso with Overlap)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test/Valid split

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Model definition
# MAGIC    - we want to do ensamble models, so we will define several different models here
# MAGIC    - logistic regression, XGBoost, Random forest

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter tuning
# MAGIC    - Grid search
# MAGIC    - Bayesian optimization

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Model training
# MAGIC    - separate models
# MAGIC    - ensamble model

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation
# MAGIC    - ROC, Gini
# MAGIC    - Lift, Gain chart
# MAGIC    - Confusion matrix, accuracy, F1 score, precision, and recall
# MAGIC    - Features with high intercorrelation

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable explanation
# MAGIC    - feature importances
# MAGIC    - shapley
# MAGIC    - partial dependence plot

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Documentation
# MAGIC    - write Evaluation and Explanation into html file

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Save final model

# COMMAND ----------


