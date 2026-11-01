# -*- coding: utf-8 -*-
"""
02_market_sql.py —— 5 条 Spark SQL 完成动漫市场画像

输入: /user/bigdata/anime/parquet/{anime, rating}
输出: /user/bigdata/anime/out/sql_q1 ... sql_q5  (CSV，供宿主端画图)
"""
import sys
from pyspark.sql import SparkSession

HDFS = "hdfs://master:9000/user/bigdata/anime"


def sep(t):
    print("\n" + "=" * 78 + "\n== " + t + "\n" + "=" * 78, flush=True)


def main():
    spark = (SparkSession.builder.appName("anime-02-market-sql")
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    anime = spark.read.parquet(f"{HDFS}/parquet/anime")
    rating = spark.read.parquet(f"{HDFS}/parquet/rating")
    anime.createOrReplaceTempView("anime")
    rating.createOrReplaceTempView("rating")

    # 每部番剧的评分行为统计（后续多条 SQL 复用）
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW rating_stat AS
        SELECT anime_id,
               COUNT(*)              AS rating_cnt,
               ROUND(AVG(rating), 3) AS avg_user_rating
        FROM rating
        GROUP BY anime_id
    """)

    # Q1 内容形态（TV/Movie/OVA…）的供给与口碑
    sep("Q1  内容形态分布与平均评分")
    q1 = spark.sql("""
        SELECT type,
               COUNT(*)                                   AS anime_cnt,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct,
               ROUND(AVG(rating), 3)                      AS avg_rating,
               SUM(members)                               AS total_members
        FROM anime
        WHERE type IS NOT NULL
        GROUP BY type
        ORDER BY anime_cnt DESC
    """)
    q1.show(truncate=False)
    q1.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/sql_q1")

    # Q2 人气 Top10（members）
    sep("Q2  人气 Top10 番剧（members）")
    q2 = spark.sql("""
        SELECT name, type, episodes, rating, members
        FROM anime
        WHERE members IS NOT NULL
        ORDER BY members DESC
        LIMIT 10
    """)
    q2.show(truncate=False)
    q2.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/sql_q2")

    # Q3 口碑 Top10（加 members >= 10000 门槛，避免"1 人打 10 分"的噪声）
    sep("Q3  口碑 Top10（members >= 10000）")
    q3 = spark.sql("""
        SELECT name, type, episodes, rating, members, rating_cnt AS user_rating_cnt
        FROM anime a
        LEFT JOIN rating_stat r USING (anime_id)
        WHERE a.members >= 10000 AND a.rating IS NOT NULL
        ORDER BY a.rating DESC, a.members DESC
        LIMIT 10
    """)
    q3.show(truncate=False)
    q3.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/sql_q3")

    # Q4 题材（genre）拆分统计 —— explode 一行多题材
    sep("Q4  题材 Top12（explode genre）")
    q4 = spark.sql("""
        SELECT genre_name,
               COUNT(*)              AS anime_cnt,
               ROUND(AVG(rating), 3) AS avg_rating,
               SUM(members)          AS total_members
        FROM (
            SELECT TRIM(g) AS genre_name, rating, members
            FROM anime
            LATERAL VIEW explode(split(genre, ',')) t AS g
            WHERE genre IS NOT NULL AND TRIM(g) <> ''
        )
        GROUP BY genre_name
        ORDER BY anime_cnt DESC
        LIMIT 12
    """)
    q4.show(truncate=False)
    q4.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/sql_q4")

    # Q5 评分分箱 + 与用户实际打分的关系
    sep("Q5  站内评分分箱分布（含用户评分行为）")
    q5 = spark.sql("""
        SELECT CASE
                 WHEN rating < 6.0 THEN 'a_<6.0'
                 WHEN rating < 7.0 THEN 'b_6.0-7.0'
                 WHEN rating < 8.0 THEN 'c_7.0-8.0'
                 WHEN rating < 9.0 THEN 'd_8.0-9.0'
                 ELSE 'e_>=9.0'
               END                     AS rating_bin,
               COUNT(*)                AS anime_cnt,
               SUM(members)            AS total_members,
               ROUND(AVG(rs.avg_user_rating), 3) AS avg_user_rating
        FROM anime a
        LEFT JOIN rating_stat rs USING (anime_id)
        WHERE a.rating IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """)
    q5.show(truncate=False)
    q5.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/sql_q5")

    sep("附加：评分行为最活跃的 10 位用户 + 最爱题材")
    extra = spark.sql("""
        SELECT r.user_id, COUNT(*) AS cnt, ROUND(AVG(r.rating), 3) AS avg_r
        FROM rating r GROUP BY r.user_id ORDER BY cnt DESC LIMIT 10
    """)
    extra.show(truncate=False)
    extra.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/sql_extra")

    spark.stop()


if __name__ == "__main__":
    sys.exit(main())
