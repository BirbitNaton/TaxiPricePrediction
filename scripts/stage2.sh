#!/bin/bash

# Move AVSC schemas to HDFS
hdfs dfs -test -d /project/avsc && hdfs dfs -rm -r /project/avsc
hdfs dfs -mkdir /project/avsc
hdfs dfs -put ./data/avsc/*.avsc /project/avsc

hive -f hql/db.hql
