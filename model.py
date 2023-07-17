# Databricks notebook source
# MAGIC %md
# MAGIC # Model template

# COMMAND ----------

# MAGIC %md
# MAGIC ## Libraries

# COMMAND ----------

!pip install unidecode

# COMMAND ----------

import pandas as pd
import numpy as np

import re
from unidecode import unidecode

from sklearn.impute import SimpleImputer
# from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import mlflow
import mlflow.sklearn

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auxiliary funtions

# COMMAND ----------

def ColumnNameNormalization(input_str):
    # Translate diacritics and remove any non-alphanumeric characters except underscores
    output_str = re.sub(r'[^a-zA-Z0-9_]', '', unidecode(input_str).lower())
    
    # Remove leading digits and add an underscore if the name starts with a digit
    if output_str and output_str[0].isdigit():
        output_str = '_' + output_str.lstrip('0123456789')
    
    # Ensure the name starts with a letter
    if output_str and not output_str[0].isalpha():
        output_str = '_' + output_str
    
    return output_str

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable definitions

# COMMAND ----------

target = 'target'

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

df = data.select("*").toPandas()

df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preparation
# MAGIC    - remove columns with all missing values
# MAGIC    - standard scaling
# MAGIC    - one hot encoding
# MAGIC    - missing imputation

# COMMAND ----------

df = df.dropna(axis=1, how='all')

Y = df.target
X = df.loc[:, df.columns != target]

technical_columns = ['pt_unified_key','edt','timestamp']
# categorical_columns = X.select_dtypes(include='object').columns.tolist()
categorical_columns = ['vzdelani', 'pohlavi', 'marital_status', 'orgh_region_name', 'ptst_city_size_cat', 'ZIVOTNI_FAZE']

numerical_columns = [x for x in X.columns if x not in categorical_columns and x not in technical_columns]

display(X)

# COMMAND ----------

Imputation = SimpleImputer(missing_values=np.nan, strategy='mean')

X_num = X[numerical_columns]
X_num_array = Imputation.fit_transform(X_num)

# COMMAND ----------

Scaler = StandardScaler()

X_num = pd.DataFrame(Scaler.fit_transform(X_num_array) , index=X_num.index, columns=X_num.columns)

display(X_num)

# COMMAND ----------

OneHot = OneHotEncoder()

X_cat = X[categorical_columns]
X_cat = OneHot.fit_transform(X_cat).toarray()

onehot_column_names = OneHot.get_feature_names_out(input_features=categorical_columns)
onehot_column_names = [ColumnNameNormalization(x) for x in onehot_column_names]

X_cat = pd.DataFrame(X_cat, columns=onehot_column_names)

display(X_cat)

# COMMAND ----------

with mlflow.start_run(run_name='MSP testing'):
    mlflow.log_param("imputer_strategy", "mean")
    mlflow.log_param("scaler_strategy", "StandardScaler")
    mlflow.log_param("encoder_strategy", "OneHotEncoder")
    
    # Log the imputer and scaler models to MLflow
    mlflow.sklearn.log_model(Imputation, "imputer_model")
    mlflow.sklearn.log_model(Scaler, "scaler_model")
    mlflow.sklearn.log_model(OneHot, "encoder_model")
    
    preprocessed_df = pd.concat([X_num, X_cat], axis=1)

# COMMAND ----------



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
