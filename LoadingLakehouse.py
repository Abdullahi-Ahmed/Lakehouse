# Databricks notebook source
# MAGIC %md 
# MAGIC # Data Path

# COMMAND ----------

#creation of the Path
projectPath     = f"/mnt/<mount-name>/Lakehouse"
bronzePath     = projectPath + "/bronze/"
silverPath      = projectPath + "/silver/"
goldPath        = projectPath + "/gold/"

# COMMAND ----------

# MAGIC %md 
# MAGIC # Create Database

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS <NameYourdb>")
spark.sql(f"USE <NameYourdb>");

# COMMAND ----------

def retrieve_data(file: str) -> bool:

    loadPath = "/mnt/<mount-name>" + file
    dbfsPath   = bronzePath + file
    dbutils.fs.cp(loadPath , dbfsPath,recurse= True)
    return True

def load_delta_table(file: str, delta_table_path: str) -> bool:
    parquet_df =spark.read.format("csv,  parquest or com.crealytics.spark.excel").option("header", "true") .option("inferSchema", "true").option("/mnt/<mount-name>", "Sheet1").load(bronzePath + file)
    parquet_df.write.format('delta').save(delta_table_path)              
    return True

def process_file(file_name: str, path: str,  table_name: str) -> bool:
  """
  1. retrieve file
  2. load as delta table
  3. register table in the metastore
  """

  retrieve_data(file_name)
  print(f"Retrieve {file_name}.")

  load_delta_table(file_name, path)
  print(f"Load {file_name} to {path}")

  spark.sql(f"""
  DROP TABLE IF EXISTS {table_name}
  """)

  spark.sql(f"""
  CREATE TABLE {table_name}
  USING DELTA
  LOCATION "{path}"
  """)

  print(f"Register {table_name} using path: {path}")

# COMMAND ----------

# Use the utility function `process_file` to retrieve the data
# Use the arguments in the table above.

process_file(
  FILL_IN_FILE_NAME,
  FILL_IN_PATH,
  FILL_IN_TABLE_NAME
)
