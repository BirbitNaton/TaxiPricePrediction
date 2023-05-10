from pyspark.sql import SparkSession

spark = SparkSession.builder\
        .appName("BDT Project")\
        .config("spark.sql.catalogImplementation","hive")\
        .config("hive.metastore.uris", "thrift://sandbox-hdp.hortonworks.com:9083")\
        .config("spark.sql.avro.compression.codec", "snappy")\
        .enableHiveSupport()\
        .getOrCreate()

trips = spark.read.format("avro").table('projectdb.trip_data')
trips.createOrReplaceTempView('trips')

geo = spark.read.format("avro").table('projectdb.zone_geo')
geo.createOrReplaceTempView('geo')

from pyspark.sql.types import StructType, StringType, IntegerType, TimestampType, FloatType, DoubleType
from pyspark.sql.functions import col, year, month, dayofmonth, udf, hour, minute, max, sin, cos, col, from_unixtime
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, VectorIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
# from matplotlib import pyplot as plt
import numpy as np
import shapely.wkt as poly
import pyproj
# from functools import partial
from shapely.geometry import shape
from shapely.ops import transform

# trip_schema = StructType() \
#     .add("vendor_id",IntegerType(),True) \
#     .add("pickup_datetime",TimestampType(),True) \
#     .add("dropoff_datetime",TimestampType(),True) \
#     .add("passenger_count",IntegerType(),True) \
#     .add("trip_distance",FloatType(),True) \
#     .add("rate_code",IntegerType(),True) \
#     .add("store_and_fwd_flag",StringType(),True) \
#     .add("payment_type",IntegerType(),True) \
#     .add("fare_amount",FloatType(),True) \
#     .add("extra",FloatType(),True) \
#     .add("mta_tax",FloatType(),True)\
#     .add("tip_amount",FloatType(),True)\
#     .add("tolls_amount",FloatType(),True) \
#     .add("imp_surcharge",FloatType(),True) \
#     .add("total_amount",FloatType(),True) \
#     .add("pickup_location_id",IntegerType(),True) \
#     .add("dropoff_location_id",IntegerType(),True)

# geo_schema = StructType() \
#     .add("zone_id",IntegerType(),True) \
#     .add("zone_name",StringType(),True) \
#     .add("borough",StringType(),True) \
#     .add("zone_geom",StringType(),True)

# trips = spark.read.csv("taxi_trip_data.csv", header = True, schema = trip_schema, multiLine=True, escape="\"")
# geo = spark.read.csv("taxi_zone_geo.csv", header = True, schema = geo_schema, multiLine=True, escape="\"")
# trips

pos_distance = (trips.trip_distance > 0)
pos_total = (trips.total_amount > 0)
pos_passengers = (trips.passenger_count > 0)
pos_fare = (trips.fare_amount > 0)
columns_needed = ["pickup_datetime", 'dropoff_datetime', 'passenger_count', 'trip_distance', 
                  'rate_code', 'payment_type', 'pickup_location_id', 'dropoff_location_id', 'total_amount']
trips = trips.withColumn("total_amount", trips.total_amount - trips.tip_amount)
trips = trips.filter((trips.trip_distance > 0) & (trips.total_amount > 0) & (trips.passenger_count > 0) & (trips.fare_amount > 0)).dropna().select(columns_needed)

trips = trips.withColumn("dropoff_month", month(from_unixtime(trips.dropoff_datetime/1000)))
trips = trips.withColumn("pickup_month", month(from_unixtime(trips.pickup_datetime/1000)))

trips.show(5)

trips = trips.withColumn("dropoff_time_sin", 2*np.pi*(hour(from_unixtime(trips.dropoff_datetime/1000))/24 + minute(from_unixtime(trips.dropoff_datetime/1000))/60/60))
trips = trips.withColumn("dropoff_time_cos", 2*np.pi*(hour(from_unixtime(trips.dropoff_datetime/1000))/24 + minute(from_unixtime(trips.dropoff_datetime/1000))/60/60))
trips = trips.withColumn("pickup_time_sin", 2*np.pi*(hour(from_unixtime(trips.pickup_datetime/1000))/24 + minute(from_unixtime(trips.pickup_datetime/1000))/60/60))
trips = trips.withColumn("pickup_time_cos", 2*np.pi*(hour(from_unixtime(trips.pickup_datetime/1000))/24 + minute(from_unixtime(trips.pickup_datetime/1000))/60/60))
trips = trips.withColumn("dropoff_time_sin", cos("dropoff_time_sin"))
trips = trips.withColumn("dropoff_time_cos", cos("dropoff_time_cos"))
trips = trips.withColumn("pickup_time_sin", cos("pickup_time_sin"))
trips = trips.withColumn("pickup_time_cos", cos("pickup_time_cos"))

columns_needed = ['passenger_count', 'trip_distance', 'rate_code', 'payment_type', 'pickup_location_id', 
                  'dropoff_location_id', 'total_amount', 'dropoff_time_sin', 'dropoff_time_cos', 'pickup_time_sin', 'pickup_time_cos']
trips = trips.select(columns_needed)

trips = trips.alias('trips').join(geo.withColumnRenamed('zone_id',"dropoff_location_id").alias('dropoff'),["dropoff_location_id"],"leftouter")
trips = trips.withColumnRenamed("zone_geom", "dropoff_polygon")
trips = trips.alias('trips').join(geo.withColumnRenamed('zone_id',"pickup_location_id").alias('dropoff'),["pickup_location_id"],"leftouter")
trips = trips.withColumnRenamed("zone_geom", "pickup_polygon")
trips = trips.select(columns_needed+["pickup_polygon", "dropoff_polygon"])
trips = trips.dropna()
trips = trips.withColumn("trip_distance", trips["trip_distance"].cast(DoubleType()))
trips.show(10)

# proj = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'),
#                 pyproj.Proj(init='epsg:3857'))
# def poly_area(polygon):
#     polygon = poly.loads(polygon)
#     return transform(proj, shape(polygon)).area
# get_area = udf(lambda x: poly_area(x), DoubleType())
# trips = trips.withColumn("dropoff_polygon", get_area("dropoff_polygon"))
# trips = trips.withColumn("pickup_polygon", get_area("pickup_polygon"))

# # preprocessed_dataset = trips #.limit(100)
# feature_list = ['passenger_count', 'trip_distance', 'rate_code', 'payment_type', 'pickup_location_id', 
#                   'dropoff_location_id', 'dropoff_time_sin', 'dropoff_time_cos', 'pickup_time_sin', 'pickup_time_cos']
# assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
# # preprocessed_dataset = assembler.transform(preprocessed_dataset)

# rfr = RandomForestRegressor(featuresCol='features', labelCol="total_amount", predictionCol="prediction")
# rfr_evaluator = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="rmse")

# pipeline = Pipeline(stages=[assembler, rfr])
# paramGrid = ParamGridBuilder() \
#     .addGrid(rfr.numTrees, list(range(10, 151, 10))) \
#     .addGrid(rfr.maxDepth, list(range(5, 26, 5))) \
#     .build()

# crossval = CrossValidator(estimator=rfr,
#                           estimatorParamMaps=paramGrid,
#                           evaluator=rfr_evaluator,
#                           numFolds=5)

# (train, test) = (train, test) = assembler.transform(trips).randomSplit([0.8, 0.2])
# clf_cv = crossval.fit(train)
# pred = clf_cv.transform(test)

# rmse = rfr_evaluator.evaluate(pred)

# bestPipeline = clf_cv.bestModel
# bestModel = bestPipeline.stages[1]

# rmse, bestModel.getNumTrees, bestModel.getOrDefault('maxDepth')

preprocessed_dataset = trips #.limit(100)
feature_list = ['passenger_count', 'trip_distance', 'rate_code', 'payment_type', 'pickup_location_id', 
                  'dropoff_location_id', 'dropoff_time_sin', 'dropoff_time_cos', 'pickup_time_sin', 'pickup_time_cos']
assembler = VectorAssembler(inputCols=feature_list, outputCol="features")

rfr = RandomForestRegressor(featuresCol='features', labelCol="total_amount", predictionCol="prediction")
rfr_evaluator = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="rmse")

pipeline = Pipeline(stages=[assembler, rfr])
paramGrid = ParamGridBuilder() \
    .addGrid(rfr.numTrees, list(range(10, 151, 10))) \
    .addGrid(rfr.maxDepth, list(range(5, 26, 5))) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=rfr_evaluator,
                          numFolds=5)

(train, test) = preprocessed_dataset.randomSplit([0.8, 0.2])
clf_cv = crossval.fit(train)
pred = clf_cv.transform(test)

rmse = rfr_evaluator.evaluate(pred)

bestPipeline = clf_cv.bestModel
bestModel = bestPipeline.stages[1]

rmse, bestModel.getNumTrees, bestModel.getOrDefault('maxDepth')

# plt.figure(figsize=(20,6))

importances = bestModel.featureImportances
x_values = list(range(len(importances)))

# plt.bar(x_values, importances, orientation = 'vertical')
# plt.xticks(x_values, feature_list)
# plt.ylabel('Importance')
# plt.xlabel('Feature')
# plt.title('Feature Importances')

# while True:
#     pass



