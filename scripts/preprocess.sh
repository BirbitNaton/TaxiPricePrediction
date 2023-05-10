#!/bin/bash
rm -rf data/taxi_trip_data.csv
cat data/taxi_trip_data_*.csv >> data/taxi_trip_data.csv