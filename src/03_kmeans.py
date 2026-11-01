# -*- coding: utf-8 -*-
"""
03_kmeans.py —— 番剧分群（K-Means）

特征: 站内评分 rating、人气 log1p(members)、篇幅 episodes
流程: 缺失填补 -> VectorAssembler -> StandardScaler -> KMeans(k=2..8 肘部) -> 取 k=4
簇命名: 统计各簇的人气、口碑、篇幅均值后，按与全体均值的相对位置划分并命名
"""
import sys
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

HDFS = "hdfs://master:9000/user/bigdata/anime"


def sep(t):
    print("\n" + "=" * 78 + "\n== " + t + "\n" + "=" * 78, flush=True)


def main():
    spark = (SparkSession.builder.appName("anime-03-kmeans")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    anime = spark.read.parquet(f"{HDFS}/parquet/anime")

    # ---- 特征工程 ----
    sep("STEP 1  特征工程")
    med = anime.approxQuantile("episodes", [0.5], 0.01)[0] or 12.0
    mean_rating = anime.agg(F.avg("rating")).first()[0]
    print(f"episodes 中位数填补值 = {med},  rating 均值填补值 = {mean_rating:.3f}")

    df = (anime
          .withColumn("episodes_f", F.least(F.col("episodes").cast("double"), F.lit(300.0)))
          .withColumn("episodes_f", F.when(F.col("episodes_f").isNull(), F.lit(float(med)))
                                      .otherwise(F.col("episodes_f")))
          .withColumn("rating_f", F.coalesce(F.col("rating"), F.lit(float(mean_rating))))
          .withColumn("members_f", F.log1p(F.coalesce(F.col("members"), F.lit(0)).cast("double")))
          .select("anime_id", "name", "type", "genre", "rating", "members",
                  "episodes_f", "rating_f", "members_f")
          .na.drop(subset=["rating_f", "members_f", "episodes_f"]))

    n = df.count()
    print(f"参与聚类的番剧数 = {n:,}")

    assembler = VectorAssembler(inputCols=["rating_f", "members_f", "episodes_f"], outputCol="feat")
    scaler = StandardScaler(inputCol="feat", outputCol="scaled", withMean=True, withStd=True)
    scaled = scaler.fit(assembler.transform(df)).transform(assembler.transform(df)).cache()

    # ---- 肘部法选 k ----
    sep("STEP 2  肘部法：k=2..8 的 WSSSE 与轮廓系数")
    evaluator = ClusteringEvaluator(featuresCol="scaled", metricName="silhouette", distanceMeasure="squaredEuclidean")
    rows = []
    for k in range(2, 9):
        m = KMeans(featuresCol="scaled", k=k, seed=42, maxIter=50).fit(scaled)
        wssse = m.summary.trainingCost
        sil = evaluator.evaluate(m.transform(scaled))
        rows.append((k, wssse, sil))
        print(f"k={k}  WSSSE={wssse:,.2f}  silhouette={sil:.4f}")
    elbow = spark.createDataFrame(rows, "k int, wssse double, silhouette double")
    elbow.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/kmeans_elbow")

    # ---- 正式模型 k=4 ----
    sep("STEP 3  K-Means(k=4) 训练与簇画像")
    km = KMeans(featuresCol="scaled", k=4, seed=42, maxIter=100).fit(scaled)
    pred = km.transform(scaled).withColumnRenamed("prediction", "cluster").cache()

    prof = (pred.groupBy("cluster").agg(
                F.count("*").alias("cnt"),
                F.round(F.avg("rating_f"), 3).alias("avg_rating"),
                F.round(F.avg("members"), 0).alias("avg_members"),
                F.round(F.avg("episodes_f"), 1).alias("avg_episodes"),
                F.round(F.sum("members"), 0).alias("sum_members"),
            ).orderBy("cluster"))
    prof.show(truncate=False)

    print("簇中心（标准化空间）:")
    for i, c in enumerate(km.clusterCenters()):
        print(f"  cluster {i}: rating={c[0]:+.3f}  log1p(members)={c[1]:+.3f}  episodes={c[2]:+.3f}")

    # ---- 命名：按簇均值相对全体均值的位置划分 ----
    sep("STEP 4  簇特征统计与命名（按人气/口碑的相对位置划分）")
    g_mean_r = pred.agg(F.avg("rating_f")).first()[0]
    g_mean_m = pred.agg(F.avg("members")).first()[0]
    print(f"全体均值: rating={g_mean_r:.3f}  members={g_mean_m:,.0f}")
    stats = {r["cluster"]: r.asDict() for r in prof.collect()}
    label_map = {}
    for cl, s in stats.items():
        hot = "高热" if s["avg_members"] >= g_mean_m else "小众"
        good = "好评" if s["avg_rating"] >= g_mean_r else "平庸"
        label_map[cl] = f"{hot}{good}"
    # 处理重名：用篇幅区分"长篇连载"与"短篇/剧场版"，阈值仍由数据决定
    dup = {}
    for cl, lb in label_map.items():
        dup.setdefault(lb, []).append(cl)
    for lb, cls in dup.items():
        if len(cls) > 1:
            ordered = sorted(cls, key=lambda c: -stats[c]["avg_episodes"])
            for i, cl in enumerate(ordered):
                ep = stats[cl]["avg_episodes"]
                if ep >= 100:
                    tag = "长篇连载"
                elif ep <= 6:
                    tag = "短篇/剧场版"
                else:
                    tag = f"{ep:.0f}集"
                label_map[cl] = f"{lb}·{tag}"
    # 极端情况兜底：仍有重名则追加簇号
    seen = {}
    for cl, lb in label_map.items():
        if lb in seen.values():
            label_map[cl] = f"{lb}#{cl}"
        seen[cl] = label_map[cl]
    print("簇 -> 自动标签: " + ", ".join(f"{k}:{v}" for k, v in sorted(label_map.items())))

    add_label = F.create_map(*[x for k in sorted(label_map)
                               for x in (F.lit(k), F.lit(label_map[k]))])
    pred2 = pred.withColumn("cluster_label", add_label[F.col("cluster")])

    sep("STEP 5  每个簇的典型成员（距簇中心最近 5 部）与人气最高 5 部")
    from pyspark.sql.window import Window
    import math
    centers = {i: list(c) for i, c in enumerate(km.clusterCenters())}

    def _dist(cluster, vec):
        c = centers[int(cluster)]
        v = list(vec)
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v, c)))

    dist_udf = F.udf(_dist, "double")
    pred3 = pred2.withColumn("dist", dist_udf(F.col("cluster"), F.col("scaled")))
    w_typ = Window.partitionBy("cluster").orderBy(F.asc("dist"))
    typical = (pred3.withColumn("rk", F.row_number().over(w_typ)).filter(F.col("rk") <= 5)
               .select("cluster", "cluster_label", "rk", "name", "type", "rating",
                       "members", "episodes_f", "dist"))
    typical.orderBy("cluster", "rk").show(40, truncate=False)

    w_hot = Window.partitionBy("cluster").orderBy(F.desc("members"))
    top = (pred3.withColumn("rk", F.row_number().over(w_hot)).filter(F.col("rk") <= 5)
           .select("cluster", "cluster_label", "rk", "name", "type", "rating", "members", "episodes_f"))
    top.orderBy("cluster", "rk").show(40, truncate=False)

    # ---- 落盘 ----
    sep("STEP 6  结果落盘")
    out_prof = prof.join(
        spark.createDataFrame([(k, v) for k, v in label_map.items()], "cluster int, cluster_label string"),
        "cluster").orderBy("cluster")
    out_prof.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/kmeans_profile")
    top.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/kmeans_top")
    typical.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/kmeans_typical")
    (pred2.select("cluster", "cluster_label", "members", "rating_f", "episodes_f")
     .coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/kmeans_points"))
    out_prof.show(truncate=False)
    spark.stop()


if __name__ == "__main__":
    sys.exit(main())
