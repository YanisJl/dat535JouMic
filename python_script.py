# This a python script for executing all the pipeline in one go.
# It is the same code as the one the jupyter notebook.


from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.rdd import PipelinedRDD
from itertools import islice
from datetime import datetime
from functools import partial
from typing import Callable, List, Any, Tuple
from pyspark.sql import Row, SparkSession
import time
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.clustering import KMeans


conf = SparkConf() \
    .setAppName("project") \
    .set("spark.master", "yarn") \
    .set("spark.deploy.mode", "cluster") \
    .set("spark.executor.instances", "3") \
    .set("spark.executor.cores", "4") \
    .set("spark.executor.memory", "2G") \

sc = SparkContext(conf=conf)

spark = SparkSession(sc)

data = sc.textFile("hdfs:///project/unstructured_dengue_100.txt") 
print(data.getNumPartitions())
def remove_quotation_marks(row: List[Any]):
        row[0] = row[0].strip("'")
        row[-1] = row[-1].strip("'")
        return row


data = data \
        .map(lambda row: row.split("' '")) \
        .map(lambda row: remove_quotation_marks(row))


selected_columns = [0, 2, 3, 4, 5, 18, 19, 20]

data = data.map(lambda row: [row[col] for col in selected_columns])

data = data.mapPartitionsWithIndex(
    lambda idx, it: islice(it, 1, None) if idx == 0 else it
)

def convert_datatypes(row: List[Any], convert_function: Callable[[Any], Any], idxs: List[int]) -> List[Any]:
    for idx in idxs:
        row[idx] = convert_function(row[idx]) if row[idx] != '' else None
    return row


def convert_to_datetime(input_str: str, format: str) -> datetime:
    return datetime.strptime(input_str, format)


datetime_format = '%Y/%m/%d'

data = data \
        .map(lambda row: convert_datatypes(row, partial(convert_to_datetime,
                                                    format=datetime_format),
                                                    [0, 1])) \
        .map(lambda row: convert_datatypes(row, int, [-1]))

data.cache()

data = data.zipWithIndex()
data = data.map(lambda row: row[0] + [row[1]])

data_with_index = data

def convert_to_timestamp_row(row: List[str], idx: int) -> List[str]:
        row[idx] = time.mktime(row[idx].timetuple())
        return row


def get_days_since_row(row: List[Any], idx: int,
                   date: int = None, date_idx: int = None) -> List[Any]:
    """
    Calculate number of days between specified date columns.

    :param row: data row
    :param idx: idx to be recalculated
    :param date: date to be subtracted, only used if date_idx is not specified
    :param date_idx: idx of the column with the date to be subtracted
    :return: new row
    """
    if row[idx] is None:
        return row

    if date_idx is not None:
        date = row[date_idx]

    if date is None:
        raise Exception('Specify date or date_idx')

    row[idx] = row[idx]- date
    return row


def convert_sex(row: List[Any], idx: int) -> List[Any]:
    """
    Convert values of "sex" column to number encoding
    """
    row[idx] = 0 if row[idx] == 'M' else 1
    return row


def convert_age(row: List[Any], idx: int) -> List[Any]:
    """
    Convert values of "age" column to number encoding
    """
    row[idx] = int(row[idx].split('-')[0].split('+')[0])
    return row


def convert_infected(row: List[Any], idx: int) -> List[Any]:
    """
    Convert values of "imported" column to number encoding
    """
    row[idx] = 0 if row[idx] == 'N' else 1
    return row


data_clustering = data_with_index \
                    .map(lambda row: convert_to_timestamp_row(row, 1)) \
                    .map(lambda row: convert_to_timestamp_row(row, 0)) \
                    .map(lambda row: get_days_since_row(row, 1, date_idx=0)) \
                    .map(lambda row: convert_sex(row, 2)) \
                    .map(lambda row: convert_age(row, 3)) \
                    .map(lambda row: convert_infected(row, 5))

data_clustering.cache()

def restrict_to_values(row: List[Any], idx: int, values: List[str]) -> List[Any]:
    """
    Assign values not present in "values" to "Other"
    """
    if row[idx] not in values:
        row[idx] = 'Other'
    return row


def reduce_number_of_categories_for_column(data: PipelinedRDD,
                                            idx: int,
                                            limit: int = 500) -> Tuple[Any, List[str]]:
    """
    Reduce number of categories in a column to only those that have occurences higher than limit
    """
    counts = data.map(lambda row: (row[idx], 1)).reduceByKey(lambda a, b: a + b)

    values_to_be_kept = (
        counts
        .map(lambda row: ('Other', row[1]) if row[1] < limit or row[0] == 'None' else row)
        .reduceByKey(lambda a, b: a + b)
        .map(lambda row: row[0])
        .collect()
    )

    data = data.map(lambda row: restrict_to_values(row, idx, values_to_be_kept))

    return data, values_to_be_kept


def one_hot(row: List[Any], idx: int, values: List[str]) -> List[Any]:
    """
    One hot encodes selected column
    """
    for value in values:
        row.append(1 if row[idx] == value else 0)

    del row[idx]
    return row


# reduce number of cathegories for counties column
data_clustering, counties_to_be_kept = reduce_number_of_categories_for_column(data_clustering, 4)

# reduce number of cathegories for countries column
data_clustering, countries_to_be_kept = reduce_number_of_categories_for_column(data_clustering, 6)

data_clustering = (
    data_clustering
    .map(lambda row: one_hot(row, 4, counties_to_be_kept))
    .map(lambda row: one_hot(row, 5, countries_to_be_kept))
)

def normalize(data: PipelinedRDD) -> PipelinedRDD:
    """
    Normalize all columns in the data to 0-1 besides the index column
    """
    max_values = (
        data
        .reduce(lambda r1, r2: [max(c1, c2) for c1, c2 in zip(r1, r2)])
        )

    return data.map(lambda row: [x / max_values[i] if i != 6 else x for i, x in enumerate(row)])


data_clustering = normalize(data_clustering)

data = data.map(lambda row: row[0])

def convert_to_timestamp(datetime_val: datetime) -> float:
    return time.mktime(datetime_val.timetuple())


data = data.map(lambda x: convert_to_timestamp(x))

data = data.map(lambda x: [x, 1])
data_regression = data.reduceByKey(lambda sum, current: sum + current)

rows = data_clustering.map(lambda x: Row(*x))
df_clustering = spark.createDataFrame(rows)

assembler = VectorAssembler(inputCols=df_clustering.columns, outputCol="features")
df = assembler.transform(df_clustering)

model = KMeans(k=10, seed=1)
model = model.fit(df)

predictions = model.transform(df)

rows_with_index = data_with_index.map(lambda x: Row(*x))
df_with_index = spark.createDataFrame(rows_with_index)
predictions = predictions.selectExpr("_7 as _9", 'prediction')
predictions = predictions.join(df_with_index, ['_9'])

predictions = predictions.selectExpr(
"_9 as Id",
"prediction",
"_1 as Date_Onset", 
"_2 as Days_To_Notification", 
"_3 as Sex", 
"_4 as Age_Group", 
"_5 as County_living", 
"_6 as Imported", 
"_7 as Country_infected", 
"_8 as Number_of_confirmed_cases")

predictions.write.mode('overwrite').parquet("hdfs:///project/clustering_results.parquet")

split_percentage = 0.90

df = spark.createDataFrame(data_regression)
df = df.sort("_1", ascending=True)
df = df.withColumnRenamed("_2", "label")

train_df_size = int(split_percentage * df.count())
df = df.sort("_1", ascending=True)
train = df.limit(train_df_size)
test = df.subtract(train)

assembler = VectorAssembler().setInputCols(['_1',]).setOutputCol('features')
train01 = assembler.transform(train)
train02 = train01.select("features","label")

lr = RandomForestRegressor()
model = lr.fit(train02)

test01 = assembler.transform(test)
test02 = test01.select('features', 'label')
regression_results = model.transform(test02)

regression_results.write.mode('overwrite').parquet("hdfs:///project/regression_results.parquet")