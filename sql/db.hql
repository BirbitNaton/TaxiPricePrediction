DROP DATABASE IF EXISTS projectdb CASCADE;
CREATE DATABASE projectdb;
USE projectdb;

CREATE EXTERNAL TABLE zone_geo STORED AS AVRO LOCATION '/project/zone_geo' TBLPROPERTIES ('avro.schema.url'='/project/zone_geo.avsc');
CREATE EXTERNAL TABLE trip_data STORED AS AVRO LOCATION '/project/trip_data' TBLPROPERTIES ('avro.schema.url'='/project/trip_data.avsc');