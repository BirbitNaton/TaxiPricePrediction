USE projectdb;
INSERT OVERWRITE LOCAL DIRECTORY '/root/q1'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT zone_name, AVG(trip_data.total_amount) AS avg_payed_in, AVG(trip_data.trip_distance) AS avg_dist_in,
       COUNT(trip_data.dropoff_location_id) AS rides_number_in,
       (AVG(trip_data.total_amount)/AVG(trip_data.trip_distance)) AS price_for_dist_in
FROM trip_data
JOIN zone_geo ON zone_geo.zone_id = trip_data.dropoff_location_id
WHERE trip_distance > 0 AND total_amount > 0
GROUP BY zone_name
ORDER BY price_for_dist_in DESC;

INSERT OVERWRITE LOCAL DIRECTORY '/root/q2'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT zone_name, AVG(trip_data.total_amount) AS avg_payed_out, AVG(trip_data.trip_distance) AS avg_dist_out,
       COUNT(trip_data.dropoff_location_id) AS rides_number_out,
       (AVG(trip_data.total_amount)/AVG(trip_data.trip_distance)) AS price_for_dist_out
FROM trip_data
JOIN zone_geo ON zone_geo.zone_id = trip_data.pickup_location_id
WHERE trip_distance > 0 AND total_amount > 0
GROUP BY zone_name
ORDER BY price_for_dist_out DESC;

INSERT OVERWRITE LOCAL DIRECTORY '/root/q3'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT dropoff_location_id, pickup_location_id, AVG(fare_amount) as fare, COUNT(*)
FROM trip_data
WHERE trip_distance > 0 AND total_amount > 0 AND passenger_count > 0 AND fare_amount > 0
GROUP BY dropoff_location_id, pickup_location_id;

INSERT OVERWRITE LOCAL DIRECTORY '/root/q4'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT passenger_count, COUNT(passenger_count),
       MIN(total_amount)/MAX(trip_distance) AS min_total_for_dist,
       AVG(total_amount)/AVG(trip_distance) AS avg_total_for_dist,
       MAX(total_amount)/MIN(trip_distance) AS max_total_for_dist
FROM trip_data
WHERE trip_distance > 0 AND total_amount > 0 AND passenger_count > 0
GROUP BY passenger_count;

INSERT OVERWRITE LOCAL DIRECTORY '/root/q6'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT month(from_unixtime(cast((pickup_datetime/1000) as bigint))) AS month, hour(from_unixtime(cast((pickup_datetime/1000) as bigint))) AS hour, COUNT(*),
       MIN(fare_amount) AS min_fare, AVG(fare_amount) AS avg_fare, MAX(fare_amount) AS max_fare
FROM trip_data
WHERE trip_distance > 0 AND total_amount > 0 AND passenger_count > 0 AND fare_amount > 0
GROUP BY month(from_unixtime(cast((pickup_datetime/1000) as bigint))), hour(from_unixtime(cast((pickup_datetime/1000) as bigint)))

