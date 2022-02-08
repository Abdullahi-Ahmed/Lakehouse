# Databricks notebook source
# MAGIC %md
# MAGIC ## Monthly AutoML Retrain

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

# Set config for database name, file paths, and table names
feature_table = 'tempdb.jiji_features'

fs = FeatureStoreClient()

features = fs.read_table(feature_table)

# COMMAND ----------

import databricks.automl
model = databricks.automl.classify(features, 
                                   target_col = "Price",
                                   data_dir= "dbfs:/tmp/omar/prediction/",
                                   timeout_minutes=120) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Promote to Registry

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

client = MlflowClient()

run_id = model.best_trial.mlflow_run_id
model_name = "RetainedModel"
model_uri = f"runs:/{run_id}/model"

client.set_tag(run_id, key='db_table', value='tempdb.jiji_features')

model_details = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Add Comments

# COMMAND ----------

model_version_details = client.get_model_version(name=model_name, version=model_details.version)

client.update_registered_model(
  name=model_details.name,
  description="This model predicts the price of a car in jiji cardealership  using features from the tempdb database we created in the begining for our class.  It is used to update the Jiji cars Dashboard and later use streamlit for web interaction ."
)

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using sklearn's linear model."
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Request Transition to Staging

# COMMAND ----------

# MAGIC %run ./helpers/registry_helpers

# COMMAND ----------

# Transition request to staging
staging_request = {'name': model_name, 'version': model_details.version, 'stage': 'Staging', 'archive_existing_versions': 'true'}
mlflow_call_endpoint('transition-requests/create', 'POST', json.dumps(staging_request))

# COMMAND ----------

# Leave a comment for the ML engineer who will be reviewing the tests
comment = "This was the best model from AutoML, I think we can use it as a baseline."
comment_body = {'name': model_name, 'version': model_details.version, 'comment': comment}
mlflow_call_endpoint('comments/create', 'POST', json.dumps(comment_body))
