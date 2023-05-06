#!/bin/bash
bash psql -U postgres -f /sql/create_db.sql
bash hdfs dfs -rm -r /project
bash sqoop import-all-tables \
    -Dmapreduce.job.user.classpath.first=true \
    --connect jdbc:postgresql://localhost/project \
    --username postgres \
    --warehouse-dir /project \
    --as-avrodatafile \
    --compression-codec=snappy \
    --outdir /project/avsc \
    --m 1
bash hdfs dfs -put /project/avsc/*.avsc /project
