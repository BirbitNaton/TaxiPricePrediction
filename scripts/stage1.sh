#!/bin/bash

# Create database and load data
psql -U postgres -f sql/create_db.sql

# Remove the files created previously
hdfs dfs -rm -r /project

# Transfer data from postgres to hdfs
sqoop import-all-tables     -Dmapreduce.job.user.classpath.first=true     --connect jdbc:postgresql://localhost/project     --username postgres     --warehouse-dir /project     --as-avrodatafile     --compression-codec=snappy     --outdir ./data/avsc     --m 1

