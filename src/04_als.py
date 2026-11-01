# -*- coding: utf-8 -*-
"""
04_als.py —— ALS 协同过滤推荐 + 超参网格搜索

数据: 有效评分 6.34M（rating 1~10），为适配 5.8G 内存，按"用户"抽样（保序：
      只保留评分条数 >= 20 的活跃用户，再按比例抽样），使每个用户的历史完整保留，
      避免按行抽样破坏用户-物品结构。

流程: 用户抽样 -> Train/Validation 8:2 -> 网格 rank{5,10,20} x regParam{0.05,0.1,0.2}
      -> 以 RMSE 选优 -> 对示例用户 recommendForUserSubset(5) 生成推荐
"""
import sys
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.window import Window

HDFS = "hdfs://master:9000/user/bigdata/anime"
SAMPLE_USERS = 8000      # 抽样用户数
MIN_RATINGS = 20         # 只保留活跃用户
RANKS = [5, 10, 20]
REGS = [0.05, 0.1, 0.2]
MAXITER = 8


def sep(t):
    print("\n" + "=" * 78 + "\n== " + t + "\n" + "=" * 78, flush=True)


def main():
    spark = (SparkSession.builder.appName("anime-04-als")
             .config("spark.sql.shuffle.partitions", "16").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    rating = spark.read.parquet(f"{HDFS}/parquet/rating").cache()
    anime = spark.read.parquet(f"{HDFS}/parquet/anime").select("anime_id", "name", "type", "rating", "members")

    sep("STEP 1  用户维度抽样（保留完整用户历史）")
    eligible = (rating.groupBy("user_id").count()
                .filter(F.col("count") >= MIN_RATINGS).select("user_id").cache())
    n_elig = eligible.count()
    frac = min(1.0, SAMPLE_USERS / max(n_elig, 1))
    users = eligible.sample(False, frac, seed=42)
    sample = rating.join(users, "user_id").cache()
    n_s = sample.count()
    n_u = sample.select("user_id").distinct().count()
    n_i = sample.select("anime_id").distinct().count()
    print(f"活跃用户（>= {MIN_RATINGS} 条评分）: {n_elig:,}，抽样比例 {frac:.4f}")
    print(f"抽样后: 用户 {n_u:,}  番剧 {n_i:,}  评分 {n_s:,} 条"
          f"（占有效评分 {n_s / 6337234:.2%}），人均 {n_s / n_u:.1f} 条")

    train, test = sample.randomSplit([0.8, 0.2], seed=42)
    train.cache(); test.cache()
    n_tr, n_te = train.count(), test.count()
    print(f"训练集 {n_tr:,} 条 / 测试集 {n_te:,} 条")

    sep("STEP 2  基线：全局均值预测")
    g_mean = train.agg(F.avg("rating")).first()[0]
    base_rmse = (test.withColumn("pred", F.lit(g_mean))
                 .select(F.sqrt(F.avg((F.col("rating") - F.col("pred")) ** 2))).first()[0])
    print(f"全局均值 = {g_mean:.4f}，基线 RMSE = {base_rmse:.4f}")

    sep("STEP 3  网格搜索 rank x regParam（共 %d 组）" % (len(RANKS) * len(REGS)))
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    grid = []
    for rank in RANKS:
        for reg in REGS:
            als = ALS(rank=rank, regParam=reg, maxIter=MAXITER, userCol="user_id",
                      itemCol="anime_id", ratingCol="rating", coldStartStrategy="drop",
                      nonnegative=False, seed=42)
            model = als.fit(train)
            pred = model.transform(test).filter(F.isnan("prediction") == False)  # noqa: E712
            rmse = evaluator.evaluate(pred)
            mae = RegressionEvaluator(metricName="mae", labelCol="rating",
                                      predictionCol="prediction").evaluate(pred)
            grid.append((rank, reg, round(rmse, 4), round(mae, 4)))
            print(f"  rank={rank:<3} reg={reg:<5} -> RMSE={rmse:.4f}  MAE={mae:.4f}")
    grid_df = spark.createDataFrame(grid, "rank int, regParam double, rmse double, mae double")
    grid_df.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/als_grid")

    best = min(grid, key=lambda r: r[2])
    print(f"\n最优参数: rank={best[0]}, regParam={best[1]}, RMSE={best[2]}, MAE={best[3]}"
          f"  (相对基线提升 {(base_rmse - best[2]) / base_rmse:.2%})")

    sep("STEP 4  用最优参数重训并生成推荐")
    als = ALS(rank=best[0], regParam=best[1], maxIter=MAXITER, userCol="user_id",
              itemCol="anime_id", ratingCol="rating", coldStartStrategy="drop", seed=42)
    model = als.fit(train)

    # 示例用户：抽样集中评分数处于 60~400 条的用户，取前 5 位（历史足够丰富且可解释）
    agg_u = (sample.groupBy("user_id").count()
             .filter(F.col("count").between(60, 400)).orderBy(F.desc("count")))
    demo_users = [int(r["user_id"]) for r in agg_u.limit(5).collect()]
    demo_df = spark.createDataFrame([(u,) for u in demo_users], "user_id int")
    print(f"示例用户: {demo_users}")

    # ---- 4.1 原始推荐：直接取预测分最高的 5 部 ----
    recs = model.recommendForUserSubset(demo_df, 50)
    recs_flat = (recs.select("user_id", F.posexplode("recommendations").alias("pos", "rec"))
                 .select("user_id", (F.col("pos") + 1).alias("rank"),
                         F.col("rec.anime_id").alias("anime_id"),
                         F.round(F.col("rec.rating"), 3).alias("score")))
    anime_meta = anime.select(F.col("anime_id").alias("aid"), "name", "type",
                              F.col("rating").alias("anime_rating"), "members")
    cand = recs_flat.join(anime_meta, recs_flat["anime_id"] == anime_meta["aid"], "left").drop("aid")
    # 去掉用户已看过的作品，避免"推荐已评分条目"
    seen = sample.filter(F.col("user_id").isin(demo_users)).select("user_id", "anime_id") \
                 .withColumn("seen", F.lit(1))
    cand = (cand.join(seen, ["user_id", "anime_id"], "left")
            .filter(F.col("seen").isNull()).drop("seen").cache())
    raw_cols = ["user_id", "rank", "anime_id", "name", "type", "anime_rating", "members", "score"]
    raw_top = cand.filter(F.col("rank") <= 5).select(*raw_cols).orderBy("user_id", "rank")
    print("原始推荐（未加任何约束）:")
    raw_top.show(30, truncate=False)

    # ---- 4.2 人气约束后重排：隐因子模型偏爱冷门条目，需用人气下界纠偏 ----
    POP_MIN = 2000
    w = Window.partitionBy("user_id").orderBy(F.desc("score"))
    pop_top = (cand.filter(F.col("members") >= POP_MIN)
               .drop("rank")
               .withColumn("rank", F.row_number().over(w)).filter(F.col("rank") <= 5)
               .select(*raw_cols).orderBy("user_id", "rank"))
    print(f"人气约束后重排（members >= {POP_MIN}）:")
    pop_top.show(30, truncate=False)

    raw_top.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/als_recs_raw")
    pop_top.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/als_recs")

    sep("STEP 5  示例用户的历史高分记录（解释推荐依据）")
    hist = (sample.filter(F.col("user_id").isin(demo_users))
            .join(anime_meta, sample["anime_id"] == anime_meta["aid"], "left").drop("aid")
            .select("user_id", "anime_id", "name", "type",
                    F.col("rating").alias("user_rating"), "anime_rating", "members"))
    w_h = Window.partitionBy("user_id").orderBy(F.desc("user_rating"), F.desc("members"))
    hist_top = (hist.withColumn("rk", F.row_number().over(w_h)).filter(F.col("rk") <= 8)
                .drop("rk").orderBy("user_id", F.desc("user_rating")))
    hist_top.show(60, truncate=False)
    hist_top.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/als_history")
    print("示例用户评分条数: " + str({r["user_id"]: r["count"] for r in agg_u.limit(5).collect()}))

    sep("STEP 6  模型规模")
    print(f"userFactors: {model.userFactors.count():,} 行 x rank {best[0]}")
    print(f"itemFactors: {model.itemFactors.count():,} 行 x rank {best[0]}")
    print(f"SUMMARY_JSON {{'users': {n_u}, 'items': {n_i}, 'ratings': {n_s}, "
          f"'best_rank': {best[0]}, 'best_reg': {best[1]}, 'best_rmse': {best[2]}, "
          f"'baseline_rmse': {round(base_rmse, 4)}}}")
    spark.stop()


if __name__ == "__main__":
    sys.exit(main())
