# Databricks notebook source
# MAGIC %md
# MAGIC <img src="https://databricks.com/wp-content/uploads/2020/04/databricks-adds-access-control-to-mlflow-model-registry_01.jpg">

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to Use the Model Registry
# MAGIC Typically, data scientists who use MLflow will conduct many experiments, each with a number of runs that track and log metrics and parameters. During the course of this development cycle, they will select the best run within an experiment and register its model with the registry.  Think of this as **committing** the model to the registry, much as you would commit code to a version control system.  
# MAGIC 
# MAGIC The registry defines several model stages: `None`, `Staging`, `Production`, and `Archived`. Each stage has a unique meaning. For example, `Staging` is meant for model testing, while `Production` is for models that have completed the testing or review processes and have been deployed to applications. 
# MAGIC 
# MAGIC Users with appropriate permissions can transition models between stages.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Promote to Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```

# COMMAND ----------

# Import libraries
import mlflow
from mlflow.tracking import MlflowClient

# Manually set parameter values
model_name = "BaselineJiji"
#to get the model_uri run this print(f"runs:/{ mlflow_run.info.run_id }/model") where you're running/tracking your run
# Register model
model_uri = f"runs:/e7f179acd9d94e6b88da5858ce3443f0/model"
model_details = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transition to Staging
# MAGIC 
# MAGIC Next, we can transition the model we just registered to "Staging".

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Staging",
    archive_existing_versions=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transition to Production
# MAGIC 
# MAGIC Next, we can transition the model we just staged to "production".

# COMMAND ----------

client = MlflowClient()

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production",
    archive_existing_versions=True
)

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
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


