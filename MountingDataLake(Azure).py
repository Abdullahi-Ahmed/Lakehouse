# Databricks notebook source
# MAGIC %md 
# MAGIC ## Mount Azure Data Lake to DBFS
# MAGIC DBFS uses the credential that you provide when you create the mount point to access the mounted Blob storage container.
# MAGIC where
# MAGIC 
# MAGIC **storage-account-name**  is the name of your Azure Blob storage account.  
# MAGIC **container-name** is the name of a container in your Azure Blob storage account.  
# MAGIC **mount-name** is a DBFS path representing where the Blob storage container or a folder inside the container (specified in source) will be mounted in DBFS.  
# MAGIC **conf-key** can be either fs.azure.account.key.<storage-account-name>.blob.core.windows.net or fs.azure.sas.<container-name>.<storage-account-name>.blob.core.windows.net  
# MAGIC dbutils.secrets.get(scope = "<scope-name>", key = "<key-name>") gets the key that has been stored as a secret in a secret scope.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an Azure Key Vault-backed secret scope using the UI
# MAGIC Go to https:// <databricks-instance>    #secrets/createScope. This URL is case sensitive; scope in createScope must be uppercase.  
# MAGIC <img width="520" alt="datashot" src="https://user-images.githubusercontent.com/85021780/152376206-00643fba-d376-4b79-b56a-f24e94f5d957.png">

# COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://<container-name>@<storage-account-name>.blob.core.windows.net",
  mount_point = "/mnt/<mount-name>",
  extra_configs = {"<conf-key>":dbutils.secrets.get(scope = "<scope-name>", key = "<key-name>")})
