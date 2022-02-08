# Databricks notebook source
# MAGIC %md
# MAGIC ### Fetch Model in Transition

# COMMAND ----------

import mlflow, json
from mlflow.tracking import MlflowClient
#from databricks.feature_store import FeatureStoreClient

client = MlflowClient()
#fs = FeatureStoreClient()

# After receiving payload from webhooks, use MLflow client to retrieve model details and lineage
try:
  registry_event = json.loads(dbutils.widgets.get('event_message'))
  model_name = registry_event['model_name']
  version = registry_event['version']
  if 'to_stage' in registry_event and registry_event['to_stage'] != 'Staging':
    dbutils.notebook.exit()
except Exception:
  model_name = 'BaselineJiji'
  version = "1"
print(model_name, version)

# Use webhook payload to load model details and run info
model_details = client.get_model_version(model_name, version)
run_info = client.get_run(run_id=model_details.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Validate prediction

# COMMAND ----------

features = spark.read.table("tempdb.silver")


# Load model as a Spark UDF
model_uri = f'models:/BaselineJiji/1'
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# Predict on a Spark DataFrame
try:
  display(features.withColumn('predictions', loaded_model(*features.columns)))
  client.set_model_version_tag(name=model_name, version=version, key="predicts", value=1)
except Exception: 
  print("Unable to predict on features.")
  client.set_model_version_tag(name=model_name, version=version, key="predicts", value=0)
  pass

# COMMAND ----------

# MAGIC %md
# MAGIC #### Signature check
# MAGIC 
# MAGIC When working with ML models you often need to know some basic functional properties of the model at hand, such as “What inputs does it expect?” and “What output does it produce?”.  The model **signature** defines the schema of a model’s inputs and outputs. Model inputs and outputs can be either column-based or tensor-based. 
# MAGIC 
# MAGIC See [here](https://mlflow.org/docs/latest/models.html#signature-enforcement) for more details.

# COMMAND ----------

if not loaded_model.metadata.signature:
  print("This model version is missing a signature.  Please push a new version with a signature!  See https://mlflow.org/docs/latest/models.html#model-metadata for more details.")
  client.set_model_version_tag(name=model_name, version=version, key="has_signature", value=0)
else:
  client.set_model_version_tag(name=model_name, version=version, key="has_signature", value=1)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Documentation 
# MAGIC Is the model documented visually and in plain english? 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Description check
# MAGIC 
# MAGIC Has the data scientist provided a description of the model being submitted?

# COMMAND ----------

# If there's no description or an insufficient number of charaters, tag accordingly
if not model_details.description:
  client.set_model_version_tag(name=model_name, version=version, key="has_description", value=0)
  print("Did you forget to add a description?")
elif not len(model_details.description) > 20:
  client.set_model_version_tag(name=model_name, version=version, key="has_description", value=0)
  print("Your description is too basic, sorry.  Please resubmit with more detail (40 char min).")
else:
  client.set_model_version_tag(name=model_name, version=version, key="has_description", value=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Artifact check
# MAGIC Has the data scientist logged supplemental artifacts along with the original model?

# COMMAND ----------

import os

# Create local directory 
local_dir = "/tmp/model_artifacts"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

# Download artifacts from tracking server - no need to specify DBFS path here
local_path = client.download_artifacts(run_info.info.run_id, "", local_dir)

# Tag model version as possessing artifacts or not
if not os.listdir(local_path):
  client.set_model_version_tag(name=model_name, version=version, key="has_artifacts", value=0)
  print("There are no artifacts associated with this model.  Please include some data visualization or data profiling.  MLflow supports HTML, .png, and more.")
else:
  client.set_model_version_tag(name=model_name, version=version, key = "has_artifacts", value = 1)
  print("Artifacts downloaded in: {}".format(local_path))
  print("Artifacts: {}".format(os.listdir(local_path)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC 
# MAGIC Here's a summary of the testing results:

# COMMAND ----------

results = client.get_model_version(model_name, version)
results.tags
