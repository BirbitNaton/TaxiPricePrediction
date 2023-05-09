DROP DATABASE IF EXISTS project;
CREATE DATABASE project;

\c project;


-- Add tables
-- zone geo table
CREATE TABLE zone_geo_temp (
    zone_id INTEGER,
    zone_name TEXT,
    borough TEXT,
    zone_geom TEXT
);

CREATE TABLE zone_geo (
    zone_id INTEGER PRIMARY KEY,
    zone_name TEXT,
    borough TEXT,
    zone_geom TEXT
);

-- trip data table
CREATE TABLE trip_data(
    vendor_id INTEGER NOT NULL,
    pickup_datetime TIMESTAMP,
    dropoff_datetime TIMESTAMP,
    passenger_count INTEGER,
    trip_distance DECIMAL,
    rate_code INTEGER,
    store_and_fwd_flag VARCHAR(1),
    payment_type INTEGER,
    fare_amount DECIMAL,
    extra DECIMAL,
    mta_tax DECIMAL,
    tip_amount DECIMAL,
    tolls_amount DECIMAL,
    imp_surcharge DECIMAL,
    total_amount DECIMAL,
    pickup_location_id INTEGER,
    dropoff_location_id INTEGER
);

COPY zone_geo_temp FROM '/data/taxi_zone_geo.csv' DELIMITER ',' CSV HEADER NULL AS 'null' ;

INSERT INTO zone_geo
SELECT DISTINCT ON (zone_id) *
FROM zone_geo_temp;

DROP TABLE zone_geo_temp;

COPY trip_data FROM '/data/taxi_trip_data.csv' DELIMITER ',' CSV HEADER NULL AS 'null';

DELETE FROM trip_data
WHERE pickup_location_id IN (
    SELECT t.pickup_location_id
    FROM trip_data t LEFT JOIN zone_geo z ON t.pickup_location_id=z.zone_id
    WHERE z.zone_id IS NULL);

DELETE FROM trip_data
WHERE dropoff_location_id IN (
    SELECT t.dropoff_location_id
    FROM trip_data t LEFT JOIN zone_geo z ON t.dropoff_location_id=z.zone_id
    WHERE z.zone_id IS NULL);

ALTER TABLE trip_data ADD CONSTRAINT fk_pickup_location_id FOREIGN KEY(pickup_location_id) REFERENCES zone_geo (zone_id) ON DELETE SET NULL;
ALTER TABLE trip_data ADD CONSTRAINT fk_dropoff_location_id FOREIGN KEY(dropoff_location_id) REFERENCES zone_geo (zone_id) ON DELETE SET NULL;