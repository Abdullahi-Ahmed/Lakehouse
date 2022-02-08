# Databricks notebook source
# MAGIC %md
# MAGIC ## Batch Inference

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Model
# MAGIC 
# MAGIC Loading as a Spark UDF to set us up for future scale.

# COMMAND ----------

import mlflow

model = mlflow.pyfunc.spark_udf(spark, model_uri="models:/BaselineJiji/staging")

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()
features = fs.read_table('tempdb.jiji_features')

# COMMAND ----------

predictions = features.withColumn('predictions', model(*features.columns))
display(predictions.select("uid", "predictions"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Write to Delta Lake

# COMMAND ----------

predictions.write.format("delta").mode("append").saveAsTable("temp.jiji_preds")
