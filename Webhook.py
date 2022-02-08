# Databricks notebook source
import mlflow
from mlflow.utils.rest_utils import http_request
import json

def client():
    return mlflow.tracking.client.MlflowClient()

host_creds = client()._tracking_client.store.get_host_creds()
host = host_creds.host
token = host_creds.token

def mlflow_call_endpoint(endpoint, method, body='{}'):
    if method == 'GET':
        response = http_request(
            host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body)
        )
    else:
        response = http_request(
            host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, json=json.loads(body)
        )
    return response.json()

# COMMAND ----------

# MAGIC %run ./utilities/Webhook

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transition Request Notification
# MAGIC 
# MAGIC First, we set up a webhook to notify us whenever a **Model Registry transition request is created**.

# COMMAND ----------

import json 

model_name = "jijimodel"

trigger_slack = json.dumps({
    "model_name": model_name,
    "events": ["TRANSITION_REQUEST_CREATED"],
    "description": "This notification triggers when a model is requested to be transitioned to a new stage.",
    "status": "ACTIVE",
    "http_url_spec": {
        "url": slack_webhook
    }
})

mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_slack)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transition Notification
# MAGIC 
# MAGIC Rather than triggering on a request, this notification will trigger a Slack message when a model is successfully transitioned to a new stage.

# COMMAND ----------

import json 

trigger_slack = json.dumps({
  "model_name": model_name,
  "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
  "description": "This notification triggers when a model is transitioned to a new stage.",
  "http_url_spec": {
    "url": slack_webhook
  }
})

mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_slack)

# COMMAND ----------

list_model_webhooks = json.dumps({"model_name": model_name})

model_webhooks = mlflow_call_endpoint("registry-webhooks/list", method = "GET", body = list_model_webhooks)
model_webhooks

# COMMAND ----------

# MAGIC %md
# MAGIC You can also **delete webhooks**.
# MAGIC 
# MAGIC You can use the below cell to delete webhooks by ID.

# COMMAND ----------

mlflow_call_endpoint(
    "registry-webhooks/delete",
    method="DELETE",
    body=json.dumps({'id': model_webhooks["webhooks"][0]["id"]})
)

# COMMAND ----------

# MAGIC %md
# MAGIC Or you can use the below cell to delete all webhooks for a specific model.

# COMMAND ----------

for webhook in model_webhooks["webhooks"]:
    mlflow_call_endpoint(
    "registry-webhooks/delete",
    method="DELETE",
    body=json.dumps({'id': webhook["id"]})
)

# COMMAND ----------

# MAGIC %md
# MAGIC And finally, verify that they're all deleted.

# COMMAND ----------

updated_model_webhooks = mlflow_call_endpoint("registry-webhooks/list", method = "GET", body = list_model_webhooks)
updated_model_webhooks

