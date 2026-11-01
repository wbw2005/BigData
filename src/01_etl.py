# -*- coding: utf-8 -*-
"""
01_etl.py —— 原始 CSV -> 清洗 -> Parquet 列式存储（落地 HDFS）

数据源（HDFS）: /user/bigdata/anime/raw/{anime.csv, rating.csv}
产出（HDFS）  : /user/bigdata/anime/parquet/anime, /user/bigdata/anime/parquet/rating

清洗规则（全部用 Spark DataFrame API，可复现）:
  anime : episodes 为 "Unknown" -> null; rating/members 为 null 的行保留但打标;
          name 含引号逗号由 CSV reader 的 quote 处理
  rating: rating = -1 表示"看过但未打分"，与 1~10 的真实评分分开; 去重 (user_id, anime_id)
"""
import sys
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType, DoubleType
)

HDFS = "hdfs://master:9000/user/bigdata/anime"


def get_spark(app="anime-01-etl"):
    return (
        SparkSession.builder.appName(app)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


ANIME_SCHEMA = StructType([
    StructField("anime_id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("genre", StringType(), True),
    StructField("type", StringType(), True),
    StructField("episodes", StringType(), True),   # 原始含 "Unknown"
    StructField("rating", DoubleType(), True),
    StructField("members", IntegerType(), True),
])

RATING_SCHEMA = StructType([
    StructField("user_id", IntegerType(), True),
    StructField("anime_id", IntegerType(), True),
    StructField("rating", IntegerType(), True),
])


def sep(title):
    print("\n" + "=" * 78 + "\n== " + title + "\n" + "=" * 78, flush=True)


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ---------------- 1. 读取原始 CSV ----------------
    sep("STEP 1  读取 HDFS 原始 CSV")
    anime_raw = (
        spark.read.option("header", True).option("multiLine", True)
        .option("escape", '"').option("mode", "PERMISSIVE")
        .schema(ANIME_SCHEMA).csv(f"{HDFS}/raw/anime.csv")
    )
    rating_raw = (
        spark.read.option("header", True).schema(RATING_SCHEMA)
        .csv(f"{HDFS}/raw/rating.csv")
    )
    n_anime_raw = anime_raw.count()
    n_rating_raw = rating_raw.count()
    print(f"anime.csv  : {n_anime_raw:,} 行")
    print(f"rating.csv : {n_rating_raw:,} 行")
    anime_raw.printSchema()
    anime_raw.show(3, truncate=40)

    # ---------------- 2. 清洗 anime ----------------
    sep("STEP 2  清洗 anime 维表")
    anime = (
        anime_raw
        .withColumn("episodes",
                    F.when(F.upper(F.trim(F.col("episodes"))) == "UNKNOWN", None)
                     .otherwise(F.col("episodes").cast(IntegerType())))
        .withColumn("type", F.trim(F.col("type")))
        .withColumn("genre", F.trim(F.col("genre")))
        # MAL 爬取时把撇号转义成了 HTML 实体（如 Gintama&#039;），这里还原成可读文本
        .withColumn("name", F.regexp_replace(F.col("name"), r"&#0?39;", "'"))
        .withColumn("name", F.regexp_replace(F.col("name"), r"&amp;", "&"))
        .withColumn("name", F.regexp_replace(F.col("name"), r"&quot;", '"'))
        .dropDuplicates(["anime_id"])
        .cache()
    )
    n_anime = anime.count()
    st = anime.select(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("anime_id").alias("distinct_id"),
        F.sum(F.when(F.col("rating").isNull(), 1).otherwise(0)).alias("rating_null"),
        F.sum(F.when(F.col("members").isNull(), 1).otherwise(0)).alias("members_null"),
        F.sum(F.when(F.col("episodes").isNull(), 1).otherwise(0)).alias("episodes_null"),
        F.sum(F.when(F.col("genre").isNull() | (F.col("genre") == ""), 1).otherwise(0)).alias("genre_null"),
    ).first()
    print(f"清洗后 anime 行数: {n_anime:,}  (去重损失 {n_anime_raw - n_anime})")
    print("字段缺失统计: " + ", ".join(f"{k}={st[k]}" for k in st.__fields__))
    print("type 取值分布:")
    anime.groupBy("type").count().orderBy(F.desc("count")).show()

    # ---------------- 3. 清洗 rating ----------------
    sep("STEP 3  清洗 rating 事实表")
    rating_dedup = rating_raw.dropDuplicates(["user_id", "anime_id"]).cache()
    n_rating = rating_dedup.count()
    n_unrated = rating_dedup.filter(F.col("rating") == -1).count()
    rating = (
        rating_dedup
        .filter((F.col("rating") >= 1) & (F.col("rating") <= 10))
        .cache()
    )
    n_valid = rating.count()
    print(f"去重后 rating 行数     : {n_rating:,}  (原始 {n_rating_raw:,}, 重复 {n_rating_raw - n_rating:,})")
    print(f"其中 rating = -1 (看过未评分): {n_unrated:,}  占比 {n_unrated / n_rating:.2%}")
    print(f"有效评分 (1~10) 行数   : {n_valid:,}")

    # ---------------- 4. 数据画像 ----------------
    sep("STEP 4  数据画像：用户/番剧规模与稀疏度")
    prof = rating.agg(
        F.countDistinct("user_id").alias("users"),
        F.countDistinct("anime_id").alias("animes"),
        F.avg("rating").alias("avg_rating"),
        F.stddev("rating").alias("std_rating"),
        F.min("rating").alias("min_r"),
        F.max("rating").alias("max_r"),
    ).first()
    users, animes = prof["users"], prof["animes"]
    print(f"用户数        : {users:,}")
    print(f"被评分番剧数  : {animes:,}  (全集 {n_anime:,})")
    print(f"评分均值/标准差: {prof['avg_rating']:.4f} / {prof['std_rating']:.4f}  (范围 {prof['min_r']}~{prof['max_r']})")
    print(f"矩阵稀疏度    : 1 - {n_valid:,}/({users:,}x{animes:,}) = {1 - n_valid / (users * animes):.6%}")

    print("每人评分数分位数:")
    per_user = rating.groupBy("user_id").count()
    qs = per_user.approxQuantile("count", [0.1, 0.5, 0.9, 0.99], 0.01)
    print(f"  P10={qs[0]}  P50={qs[1]}  P90={qs[2]}  P99={qs[3]}  max={per_user.agg(F.max('count')).first()[0]}")
    print("评分值分布:")
    hist = rating.groupBy("rating").count().orderBy("rating")
    hist.show()
    hist.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/rating_hist")

    # ---------------- 5. 写 Parquet ----------------
    sep("STEP 5  写出 Parquet 到 HDFS")
    (anime.repartition(2).write.mode("overwrite")
     .parquet(f"{HDFS}/parquet/anime"))
    (rating.repartition(8).write.mode("overwrite")
     .parquet(f"{HDFS}/parquet/rating"))
    chk = spark.read.parquet(f"{HDFS}/parquet/anime").count()
    chr_ = spark.read.parquet(f"{HDFS}/parquet/rating").count()
    print(f"回读校验: anime={chk:,}  rating={chr_:,}")

    # 结果摘要（供报告引用）
    summary = {
        "anime_rows": n_anime,
        "rating_rows_raw": n_rating_raw,
        "rating_rows_dedup": n_rating,
        "rating_rows_valid": n_valid,
        "users": users,
        "animes_rated": animes,
        "avg_rating": round(prof["avg_rating"], 4),
        "sparsity": round(1 - n_valid / (users * animes), 8),
    }
    print("SUMMARY_JSON " + str(summary))
    spark.stop()


if __name__ == "__main__":
    sys.exit(main())
