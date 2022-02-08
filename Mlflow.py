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
model_name = "jijimodel"

runs:/d62302462ec04a16b67ad875e3e200ba/model
# Register model
model_uri = f"runs:/e1ddb78acbc94263a4a2c02d5d259377/model"
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
