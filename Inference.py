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

model = mlflow.pyfunc.spark_udf(spark, model_uri="models:/hhar_churn/staging")

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()
features = fs.read_table('ibm_telco_churn.churn_features')

# COMMAND ----------

predictions = features.withColumn('predictions', model(*features.columns))
display(predictions.select("customerId", "predictions"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Write to Delta Lake

# COMMAND ----------

predictions.write.format("delta").mode("append").saveAsTable("ibm_telco_churn.churn_preds")
