# Databricks notebook source
df =spark.read.format("com.crealytics.spark.excel").option("header", "true").option("inferSchema", "true").option("/mnt/stgcontainer/", "Sheet1").load("dbfs:/mnt/stgcontainer/JijiCarsRawDataFinal.xlsx")

# COMMAND ----------

(df.write
 .format("delta")
 .mode("overwrite")
 .saveAsTable("tempdb.bronze")
 
)

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
df = spark.read.table("tempdb.bronze")
df = (df.withColumn('Area', split(df['Location'], ',').getItem(1))
        .withColumn('City', split(df['Location'], ',').getItem(0))
        .withColumn('Mileage', regexp_replace('Mileage', 'Unavailable', '0'))
        .withColumn("Mileage",col("Mileage").cast("int")))

# COMMAND ----------

from pyspark.sql.functions import col, sha2, concat
import pandas as pd
import numpy as np

df = df.withColumn("uid", sha2(concat(col("Model"), col("Make"), col("YOM"), col("Color"), col("Used"), col("Transmission"), col("Mileage"), col("Price"), col("Area"), col("City")), 256))

df = df.toPandas()
allowed_vals = ['White', 'Black', 'Silver', 'Pearl', 'Red', 'Blue', 'Gray','Burgandy', 'Gold', 'Purple', 'Brown', 'Green', 'Orange', 'Yellow', 'Beige', 'Pink']

df['Color'] = np.where(df['Color'].isin(allowed_vals), df['Color'], 'Other')
df = df.dropna()

# COMMAND ----------

df_cleaned = spark.createDataFrame(df)
cleaned_df= df_cleaned.select(['Model','Make','YOM','Color','Used','Transmission','Mileage','Price','Area','City', 'uid'])



(cleaned_df.write
 .format("delta")
 .mode("overwrite")
 .saveAsTable("tempdb.silver")
 
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Store

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer
def compute_customer_features(data):

    categoricalCols = [field for (field, dataType) in data.dtypes 
                       if dataType == "string"]
    indexOutputCols = [x + "Index" for x in categoricalCols]
    oheOutputCols = [x + "OHE" for x in categoricalCols]

    stringIndexer = StringIndexer(inputCols=categoricalCols, 
                                  outputCols=indexOutputCols, 
                                  handleInvalid="skip")
    oheEncoder = OneHotEncoder(inputCols=indexOutputCols, 
                               outputCols=oheOutputCols)
    return data



# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

jiji_features_df = compute_customer_features(cleaned_df)

jiji_feature_table = fs.create_feature_table(
  name='tempdb.jiji_features',
  keys='uid',
  schema=jiji_features_df.spark.schema(),
  description='These features are of jiji car data table in the lakehouse.cleaned up their names there is No aggregations were performed.'
)


# COMMAND ----------

loaded_df = spark.read.table("tempdb.silver").toPandas()

# COMMAND ----------

loaded_df = spark.createDataFrame(loaded_df)
golddata= loaded_df.select(['Model','Make','YOM','Color','Used','Transmission','Mileage','Price','Area','City'])
(golddata.write
 .format("delta")
 .mode("overwrite")
 .saveAsTable("tempdb.gold")
 
)

# COMMAND ----------

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.linear_model import SGDRegressor

transformers = []


numerical_pipeline = Pipeline(steps=[
("converter", FunctionTransformer(lambda loaded_df: loaded_df.apply(pd.to_numeric, errors="coerce"))),
("imputer", SimpleImputer(strategy="mean"))
])
transformers.append(("numerical", numerical_pipeline, ["Mileage", "YOM"]))


one_hot_encoder = OneHotEncoder(handle_unknown="ignore")

transformers.append(("onehot", one_hot_encoder, ["Make", "Used", "City", "Color", "Transmission", "Model", "Area"]))

    

sgdr_regressor = SGDRegressor(
    alpha=1.9007467230037638e-06,
    average=False,
    early_stopping=True,
    fit_intercept=True,
    eta0=4.628277257193867e-06,
    learning_rate="adaptive",
    epsilon=2.9009617538355744e-05,
    loss="squared_epsilon_insensitive",
    n_iter_no_change=5,
    penalty="l1",
    tol=0.0004670369574901385,
    validation_fraction=0.1,
    random_state=865401075
)
    
    
    
model = Pipeline([
            ("preprocessor",
             ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)),
            ("standardizer", StandardScaler()),
            ("regressor", sgdr_regressor),
    ])


# COMMAND ----------

from sklearn.model_selection import train_test_split

split_X = loaded_df.drop(["Price", "uid"], axis=1)
split_y = loaded_df["Price"].copy()

# Split out train data
X_train, split_X_rem, y_train, split_y_rem = train_test_split(split_X, split_y, train_size=0.6, random_state=865401075)

# Split remaining data equally for validation and test
X_val, X_test, y_val, y_test = train_test_split(split_X_rem, split_y_rem, test_size=0.5, random_state=865401075)

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
import mlflow
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="BaselineModel") as mlflow_run:
    model.fit(X_train, y_train)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    sgdr_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

    # Log metrics for the test set
    sgdr_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

    # Display the logged metrics
    sgdr_val_metrics = {k.replace("val_", ""): v for k, v in sgdr_val_metrics.items()}
    sgdr_test_metrics = {k.replace("test_", ""): v for k, v in sgdr_test_metrics.items()}
    display(pd.DataFrame([sgdr_val_metrics, sgdr_test_metrics], index=["validation", "test"]))

# COMMAND ----------

print(f"runs:/{ mlflow_run.info.run_id }/model")

# COMMAND ----------

shap_enabled = True
if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    sample = spark.read.table("tempdb.silver").sample(0.008, seed=42).toPandas()
    data = sample.drop(["Price", "uid"], axis=1)

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_val.sample(n=1)

    # Use Kernel SHAP to explain feature importance on the example from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, data, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example)

# COMMAND ----------

# Print the absolute model uri path to the logged artifact
# Use mlflow.pyfunc.load_model(<model-uri-path>) to load this model in any notebook
print(f"Model artifact is logged at: { mlflow_run.info.artifact_uri}/model")

# COMMAND ----------


