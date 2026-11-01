# -*- coding: utf-8 -*-
"""
05_regression.py —— 用"篇幅 + 人气 + 类型"线性回归预测番剧站内评分（课程第 12 章线性回归）

目的不是做出高精度预测，而是回答一个分析问题：
    一部番剧的口碑，能否被它的篇幅、人气、类型线性解释？
结论（见 R²）：几乎不能 —— 这本身是重要发现，说明口碑信号更多来自内容质量而非规模。
同时用 GBTRegressor 做非线性对照，验证"线性模型是不是被设定限制住了"。
"""
import sys
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

HDFS = "hdfs://master:9000/user/bigdata/anime"


def sep(t):
    print("\n" + "=" * 78 + "\n== " + t + "\n" + "=" * 78, flush=True)


def main():
    spark = (SparkSession.builder.appName("anime-05-regression")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    anime = spark.read.parquet(f"{HDFS}/parquet/anime")

    sep("STEP 1  建模样本与特征")
    df = (anime
          .withColumn("episodes_f", F.least(F.col("episodes").cast("double"), F.lit(300.0)))
          .withColumn("log_members", F.log1p(F.coalesce(F.col("members"), F.lit(0)).cast("double")))
          .withColumn("genre_cnt", F.size(F.split(F.col("genre"), ",")))
          .withColumn("type_f", F.coalesce(F.col("type"), F.lit("Unknown")))
          .filter(F.col("rating").isNotNull() & F.col("episodes_f").isNotNull()))
    n = df.count()
    print(f"样本量 = {n:,} 部番剧（站内评分非空且篇幅已知）")
    df.groupBy("type_f").count().orderBy(F.desc("count")).show()

    print("与目标的相关性（Pearson）:")
    for c in ["log_members", "episodes_f", "genre_cnt"]:
        r = df.stat.corr("rating", c)
        print(f"  corr(rating, {c:<12}) = {r:+.4f}")

    sep("STEP 2  特征编码 + 线性回归")
    idx = StringIndexer(inputCol="type_f", outputCol="type_idx", handleInvalid="keep")
    ohe = OneHotEncoder(inputCols=["type_idx"], outputCols=["type_vec"], dropLast=True)
    num_cols = ["log_members", "episodes_f", "genre_cnt"]
    assembler = VectorAssembler(inputCols=num_cols + ["type_vec"], outputCol="raw_feat")
    scaler = StandardScaler(inputCol="raw_feat", outputCol="features", withMean=True, withStd=True)
    lr = LinearRegression(featuresCol="features", labelCol="rating",
                          regParam=0.01, elasticNetParam=0.0, maxIter=100)
    pipe = Pipeline(stages=[idx, ohe, assembler, scaler, lr])

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    print(f"训练 {train.count():,} / 测试 {test.count():,}")
    model = pipe.fit(train)
    lr_model = model.stages[-1]

    ev_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating")
    ev_r2 = RegressionEvaluator(metricName="r2", labelCol="rating")
    ev_mae = RegressionEvaluator(metricName="mae", labelCol="rating")
    pred = model.transform(test)
    lr_rmse, lr_r2, lr_mae = ev_rmse.evaluate(pred), ev_r2.evaluate(pred), ev_mae.evaluate(pred)
    print(f"LinearRegression: RMSE={lr_rmse:.4f}  MAE={lr_mae:.4f}  R2={lr_r2:.4f}")
    print(f"截距 = {lr_model.intercept:.4f}")
    names = num_cols + [f"type_{lb}" for lb in model.stages[0].labels]
    print("系数（标准化特征空间）:")
    for nm, co in zip(names, lr_model.coefficients):
        print(f"  {nm:<20} {co:+.4f}")

    sep("STEP 3  非线性对照：GBTRegressor")
    gbt = GBTRegressor(featuresCol="features", labelCol="rating", maxIter=60,
                       maxDepth=4, stepSize=0.1, seed=42)
    pipe_gbt = Pipeline(stages=[idx, ohe, assembler, scaler, gbt])
    gbt_model = pipe_gbt.fit(train)
    gbt_stage = gbt_model.stages[-1]
    pred_g = gbt_model.transform(test)
    g_rmse, g_r2, g_mae = ev_rmse.evaluate(pred_g), ev_r2.evaluate(pred_g), ev_mae.evaluate(pred_g)
    print(f"GBTRegressor:     RMSE={g_rmse:.4f}  MAE={g_mae:.4f}  R2={g_r2:.4f}")

    print("\nGBT 特征重要度（数值特征 + 类型独热）:")
    for nm, imp in sorted(zip(names, gbt_stage.featureImportances.toArray()),
                          key=lambda x: -x[1])[:8]:
        print(f"  {nm:<20} {imp:.4f}")

    sep("STEP 4  结论与落盘")
    spread = df.agg(F.stddev("rating")).first()[0]
    print(f"目标标准差 = {spread:.4f}，线性模型 RMSE = {lr_rmse:.4f}，R2 = {lr_r2:.4f}")
    print(f"=> 线性模型解释了 {max(lr_r2, 0):.1%} 的番剧评分方差；"
          f"若只用均值预测，RMSE 会退化到 {spread:.4f}")
    print("结论：人气(members) 与题材数量是口碑的主要可解释来源（正相关），"
          "篇幅几乎无解释力；剩余方差来自剧本/作画/演出等未观测量，"
          "说明'口碑'不能仅由规模和篇幅线性外推。")

    metrics = [("LinearRegression", round(lr_rmse, 4), round(lr_mae, 4), round(lr_r2, 4)),
               ("GBTRegressor", round(g_rmse, 4), round(g_mae, 4), round(g_r2, 4))]
    spark.createDataFrame(metrics, "model string, rmse double, mae double, r2 double") \
        .coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/reg_metrics")
    coef = [(nm, round(float(c), 4)) for nm, c in zip(names, lr_model.coefficients)]
    spark.createDataFrame(coef, "feature string, coef double") \
        .coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/reg_coef")
    imp_rows = [(nm, round(float(v), 4)) for nm, v in
                zip(names, gbt_stage.featureImportances.toArray())]
    spark.createDataFrame(imp_rows, "feature string, importance double") \
        .coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HDFS}/out/reg_importance")

    corr = [(c, round(df.stat.corr("rating", c), 4)) for c in num_cols]
    print("SUMMARY_JSON " + str({
        "n": n, "lr_rmse": round(lr_rmse, 4), "lr_r2": round(lr_r2, 4),
        "gbt_rmse": round(g_rmse, 4), "gbt_r2": round(g_r2, 4),
        "std_rating": round(spread, 4), "corr": dict(corr)}))
    spark.stop()


if __name__ == "__main__":
    sys.exit(main())
