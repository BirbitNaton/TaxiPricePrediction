#!/bin/bash
hive -f /sql/EDA.hql
rm -rf output/q1.csv
rm -rf output/q2.csv
rm -rf output/q3.csv
rm -rf output/q4.csv
rm -rf output/q6.csv
echo "zone_name,avg_payed_in,avg_dist_in,rides_number_in,price_for_dist_in" > output/q1.csv
echo "zone_name,avg_payed_out,avg_dist_out,rides_number_out,price_for_dist_out" > output/q2.csv
echo "dropoff_location_id,pickup_location_id,fare,count" > output/q3.csv
echo "passenger_count,count,min_total_for_dist,avg_total_for_dist,max_total_for_dist" > output/q4.csv
echo "month,hour,count,min_fare,avg_fare,max_fare" > output/q6.csv
cat root/q1/* >> output/q1.csv
cat root/q2/* >> output/q2.csv
cat root/q3/* >> output/q3.csv
cat root/q4/* >> output/q4.csv
cat root/q6/* >> output/q6.csv
pip install pyproj
pip install shapely
spark-submit --jars /usr/hdp/current/hive-client/lib/hive-metastore-1.2.1000.2.6.5.0-292.jar,/usr/hdp/current/hive-client/lib/hive-exec-1.2.1000.2.6.5.0-292.jar --packages org.apache.spark:spark-avro_2.12:3.0.3 scripts/bigdataproject.py