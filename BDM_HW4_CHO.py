from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
import numpy as np
import datetime
import json
import sys


def main(sc, spark):
    """
    Transfer our code from the notebook here, however, remember to replace
    the file paths with the ones provided in the problem description.
    """

    dfPlaces = spark.read.csv(
        "hdfs:///data/share/bdm/core-places-nyc.csv", header=True, escape='"'
    )
    dfPattern = spark.read.csv(
        "hdfs:///data/share/bdm/weekly-patterns-nyc-2019-2020/*",
        header=True,
        escape='"',
    )

    if len(sys.argv) > 1:
        OUTPUT_PREFIX = sys.argv[1]
    else:
        OUTPUT_PREFIX = "OUTPUT_PREFIX"

    STORE_GROUPS = {
        "Big Box Grocers": {"452210", "452311"},
        "Convenience Stores": {"445120"},
        "Drinking Places": {"722410"},
        "Full-Service Restaurants": {"722511"},
        "Limited-Service Restaurants": {"722513"},
        "Pharmacies and Drug Stores": {"466110", "446191"},
        "Snack and Bakeries": {"311811", "722515"},
        "Specialty Food Stores": {
            "445210",
            "445220",
            "445230",
            "445291",
            "445292",
            "445299",
        },
        "Supermarkets (except Convenience Stores)": {"445110"},
    }

    tmp = STORE_GROUPS.values()
    CAT_GROUP = {code: index for index, codes in enumerate(tmp) for code in codes}
    CAT_CODES = set.union(*tmp)
    CAT_LABEL = list(STORE_GROUPS.keys())

    udfToGroup = F.udf(lambda x: CAT_GROUP[x], T.IntegerType())

    dfStores = (
        dfPlaces.filter(dfPlaces["naics_code"].isin(CAT_CODES))
        .select("placekey", udfToGroup("naics_code").alias("group"))
        .cache()
    )

    groupCount = dict(dfStores.groupBy("group").count().sort("group").collect())

    def expandVisits(date_range_start, visits_by_day):
        start = datetime.datetime(*map(int, date_range_start[:10].split("-"))).date()
        for days, visits in enumerate(json.loads(visits_by_day)):
            if visits == 0:
                continue
            d = start + datetime.timedelta(days)
            if d.year in (2019, 2020):
                yield (d.year, f"{d.month:02d}-{d.day:02d}", visits)

    visitType = T.StructType(
        [
            T.StructField("year", T.IntegerType()),
            T.StructField("date", T.StringType()),
            T.StructField("visits", T.IntegerType()),
        ]
    )

    udfExpand = F.udf(expandVisits, T.ArrayType(visitType))

    def computeStats(group, visits):
        visits = np.fromiter(visits, np.int_)
        visits.resize(groupCount[group])
        median = np.median(visits)
        std_dev = np.std(visits)
        return (
            int(median + 0.5),
            max(0, int(median - std_dev + 0.5)),
            int(median + std_dev + 0.5),
        )

    statsType = T.StructType(
        [
            T.StructField("median", T.IntegerType()),
            T.StructField("low", T.IntegerType()),
            T.StructField("high", T.IntegerType()),
        ]
    )

    udfComputeStats = F.udf(computeStats, statsType)

    dfVisits = (
        (
            dfPattern.join(dfStores, "placekey")
            .withColumn(
                "expanded", F.explode(udfExpand("date_range_start", "visits_by_day"))
            )
            .select("group", "expanded.*")
        )
        .groupBy("group", "year", "date")
        .agg(F.collect_list("visits").alias("visits"))
        .withColumn("stats", udfComputeStats("group", "visits"))
        .select("group", "year", "date", "stats.*")
        .orderBy("group", "year", "date")
        .withColumn("date", F.concat(F.lit("2020-"), F.col("date")))
        .cache()
    )

    toFileName = lambda x: "_".join(
        ("".join(char if char.isalnum() else " " for char in x.lower())).split()
    )
    for index, fileName in enumerate(map(toFileName, CAT_LABEL)):
        dfVisits.filter(F.col("group") == index).drop("group").coalesce(1).write.csv(
            f"{OUTPUT_PREFIX}/{fileName}", mode="overwrite", header=True
        )


if __name__ == "__main__":
    sc = SparkContext()
    spark = SparkSession(sc)
    main(sc, spark)
