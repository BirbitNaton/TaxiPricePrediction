from pyspark.sql import SparkSession

from pyspark.sql import SparkSession

spark = SparkSession.builder\
        .appName("BDT Project")\
        .config("spark.sql.catalogImplementation","hive")\
        .config("hive.metastore.uris", "thrift://sandbox-hdp.hortonworks.com:9083")\
        .config("spark.sql.avro.compression.codec", "snappy")\
        .enableHiveSupport()\
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

trips = spark.read.format("avro").table('projectdb.trip_data')
trips.createOrReplaceTempView('trips')

geo = spark.read.format("avro").table('projectdb.zone_geo')
geo.createOrReplaceTempView('geo')

from pyspark.sql.types import StructType, StringType, IntegerType, TimestampType, FloatType, DoubleType
from pyspark.sql.functions import col, year, month, dayofmonth, udf, hour, minute, max, sin, cos, col, from_unixtime
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, VectorIndexer
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import numpy as np
import shapely.wkt as poly
import pyproj
from shapely.geometry import shape
from shapely.ops import transform

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


preprocessed_dataset = trips.limit(20000)
feature_list = ['passenger_count', 'trip_distance', 'rate_code', 'payment_type', 'pickup_location_id', 
                  'dropoff_location_id', 'dropoff_time_sin', 'dropoff_time_cos', 'pickup_time_sin', 'pickup_time_cos']
assembler = VectorAssembler(inputCols=feature_list, outputCol="features")

rfr = RandomForestRegressor(featuresCol='features', labelCol="total_amount", predictionCol="prediction")
rmse_evaluator = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="rmse")
r2_evaluator = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="r2")

pipeline = Pipeline(stages=[assembler, rfr])
paramGrid = ParamGridBuilder() \
    .addGrid(rfr.numTrees, [5, 15, 25]) \
    .addGrid(rfr.maxDepth, [5, 8, 12]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=rmse_evaluator,
                          numFolds=4)

(train, test) = preprocessed_dataset.randomSplit([0.7, 0.3])
clf_cv = crossval.fit(train)
pred = clf_cv.transform(test)

rmse = rmse_evaluator.evaluate(pred)
r2 = r2_evaluator.evaluate(pred)

bestPipeline = clf_cv.bestModel
bestModel = bestPipeline.stages[1]
clf_cv.bestModel.write().overwrite().save("model/random_forest_model")

pred.coalesce(1)\
    .select("prediction",'total_amount')\
    .write\
    .mode("overwrite")\
    .format("csv")\
    .option("sep", ",")\
    .option("header","true")\
    .csv("output/random_forest_predictions.csv")

print("Metrics. RMSE: "+str(rmse)+", R^2: "+str(r2))

gbtr = GBTRegressor(featuresCol='features', labelCol="total_amount", predictionCol="prediction")

pipeline = Pipeline(stages=[assembler, gbtr])
paramGrid = ParamGridBuilder() \
    .addGrid(gbtr.stepSize , [0.1, 0.01, 0.05]) \
    .addGrid(gbtr.maxDepth, [3, 5, 8]) \
    .addGrid(gbtr.lossType ,  ['squared', 'absolute']) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=rmse_evaluator,
                          numFolds=4)

(train, test) = preprocessed_dataset.randomSplit([0.7, 0.3])
clf_cv = crossval.fit(train)
pred = clf_cv.transform(test)

rmse = rmse_evaluator.evaluate(pred)
r2 = r2_evaluator.evaluate(pred)

bestPipeline = clf_cv.bestModel
bestModel = bestPipeline.stages[1]

pred.coalesce(1)\
    .select("prediction",'total_amount')\
    .write\
    .mode("overwrite")\
    .format("csv")\
    .option("sep", ",")\
    .option("header","true")\
    .csv("output/GBT_predictions.csv")
    
clf_cv.bestModel.write().overwrite().save("model/GBT_model")

print("Metrics. RMSE: "+str(rmse)+", R^2: "+str(r2))

