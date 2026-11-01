# 基于 Spark on YARN 的动漫评分数据分析与个性化推荐系统

《大数据技术》课程期末大作业 · 吴博文 / 523031910143 · 独立完成

在自建的 3 节点 Hadoop 集群（Ubuntu 24.04 + Docker：1 master + 2 worker）上，用
**Spark 3.5.3 on YARN** 对 MyAnimeList 的 **12,294 部番剧 / 7,813,737 条评分** 做存储、清洗、
统计分析、聚类、协同过滤与回归归因，结果导出为 CSV 后用 Matplotlib 绘图，并整理成汇报 PPT。

---

## 1. 运行环境

| 层次 | 组件 | 版本 |
|---|---|---|
| 操作系统 | Ubuntu 虚拟机（VMware） | 24.04 |
| 容器 | Docker + Docker Compose（master / worker1 / worker2） | 2 vCPU / 5.9 GB |
| 分布式存储 | HDFS（1 NameNode + 2 DataNode，副本因子 2） | Hadoop 3.3.6 |
| 资源调度 | YARN（1 ResourceManager + 2 NodeManager） | Hadoop 3.3.6 |
| 大数据计算 | Spark Core / Spark SQL | Spark 3.5.3 |
| 机器学习 | Spark MLlib（KMeans / ALS / LinearRegression / GBTRegressor） | Spark 3.5.3 |
| 数据仓库 | Hive 外部表（OpenCSVSerde） | Hive 4.0.1 |

受虚拟机 5.9 GB 内存限制，所有作业统一用
`--executor-memory 900m --executor-cores 1 --num-executors 2` 提交。

## 2. 数据获取

数据来自 Kaggle 公开数据集 *Anime Recommendations Database*（原始数据源为 MyAnimeList）：

<https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database>

在浏览器中下载后解压，把 `anime.csv` 和 `rating.csv` 放到本目录的 `data/` 下即可
（约 100 MB，未纳入仓库）。

- `anime.csv`：12,294 行 × 7 列 —— `anime_id, name, genre, type, episodes, rating, members`
- `rating.csv`：7,813,737 行 × 3 列 —— `user_id, anime_id, rating`（`-1` 表示看过但未评分）

## 3. 项目结构

```
final/
├─ src/                    PySpark 作业（全部以 spark-submit --master yarn 运行）
│  ├─ 01_etl.py            CSV → Parquet：去重、剔除 -1 分、字符还原、缺失值处理
│  ├─ 02_market_sql.py     5 条 Spark SQL 市场画像查询 + 活跃用户统计
│  ├─ 03_kmeans.py         番剧分群：k=2..8 的 WSSSE 与轮廓系数，取 k=4 建模
│  ├─ 04_als.py            ALS 协同过滤：9 组参数网格 + Top-5 推荐
│  └─ 05_regression.py     评分归因：LinearRegression 与 GBTRegressor 对照
├─ charts/make_figures.py  读取 results/*.csv，用 Matplotlib 绘制 fig1~fig9
├─ results/                各作业输出的统计结果 CSV
├─ figures/                汇报用图表 PNG（200 dpi）
└─ requirements.txt        绘图用 Python 依赖
```

## 4. 运行步骤

```bash
# ① 启动集群（虚拟机内）
bash start-hadoop.sh        # 启动 HDFS 与 YARN，确认 2 个 DataNode、2 个 NodeManager
docker ps                   # 应有 master / worker1 / worker2 三个容器

# ② 原始数据上传到 HDFS
hdfs dfs -mkdir -p /user/bigdata/anime/raw
hdfs dfs -put data/anime.csv data/rating.csv /user/bigdata/anime/raw/

# ③ 逐个提交 Spark 作业（YARN client 模式，工作目录在容器 master:/opt）
spark-submit --master yarn --executor-memory 900m --executor-cores 1 \
    --num-executors 2 src/01_etl.py
spark-submit --master yarn --executor-memory 900m --executor-cores 1 \
    --num-executors 2 src/02_market_sql.py
spark-submit --master yarn --executor-memory 900m --executor-cores 1 \
    --num-executors 2 src/03_kmeans.py
spark-submit --master yarn --executor-memory 900m --executor-cores 1 \
    --num-executors 2 src/04_als.py
spark-submit --master yarn --executor-memory 900m --executor-cores 1 \
    --num-executors 2 src/05_regression.py

# ④ 取出结果并绘图（结果目录在 HDFS 的 /user/bigdata/anime/out）
hdfs dfs -getmerge /user/bigdata/anime/out/sql_q1 ./sql_q1.csv   # 其余输出同理
python charts/make_figures.py                                     # → figures/*.png
```

## 5. 主要结果

**数据清洗**：原始 7,813,737 条评分去重后保留 7,813,730 条，剔除 `rating = -1`
（看过未打分）1,476,496 条（18.90%），最终有效评分 **6,337,234 条**，覆盖 69,600 位用户、
9,927 部作品，用户-番剧评分矩阵稀疏度 99.08%。

**市场画像（Spark SQL）**

| 问题 | 结论 |
|---|---|
| 形态结构 | TV 3,787 部（30.87%）/ OVA 26.99% / 剧场版 19.14%；TV 均分 6.90 最高，剧场版仅 6.32 |
| 人气 Top10 | Death Note 1,013,917 关注居首，Toradora! 633,817 列第十 |
| 口碑 Top10 | 设 `members ≥ 10000` 门槛后：你的名字。9.37、钢炼 FA 9.26、银魂° 9.25 |
| 题材 Top12 | Comedy 供给最多（4,645 部）但均分仅 6.62；Shounen 均分 7.06 最高 |
| 评分分箱 | <6.0 共 3,359 部；8~9 分 603 部却贡献最多关注；≥9.0 仅 24 部 |

**K-Means 分群（k=4）**：WSSSE 25,980 → 9,201，相邻降幅 50.7% → 28.2% → 18.8% → 12.8%，
k=4 之后边际收益明显转弱。训练完成后按预测簇号统计各簇的作品数、平均关注、平均评分与
平均集数，与全体均值（rating 6.474 / members 18,071）对比，归纳为四类：

| 簇 | 数量 | 平均关注 | 平均评分 | 平均集数 | 类别 |
|---|---|---|---|---|---|
| C2 | 3,750 | 54,996 | 7.41 | 13.2 | 高关注高口碑季番 |
| C0 | 5,267 | 1,814 | 6.55 | 10.8 | 中小众普通番剧 |
| C1 | 411 | 12,750 | 6.34 | 289.4 | 超长篇连载动画 |
| C3 | 2,866 | 397 | 5.12 | 4.5 | 冷门低分短片 / OVA |

**ALS 推荐**：先筛出评分 ≥20 条的活跃用户共 47,153 人，再抽样 7,953 位用户 / 8,269 部作品 /
1,046,827 条评分（占有效评分 16.52%），按 8:2 划分（训练 837,349 / 测试 209,478）。
9 组网格中最优为 `rank=20, regParam=0.1`：

| 指标 | 基线（训练集全局均值） | 最优 ALS | 提升 |
|---|---|---|---|
| RMSE | 1.5692 | **1.1500** | −26.71% |
| MAE | — | 0.8818 | — |

**评分归因（回归）**：样本 12,064 部，训练 9,717 / 测试 2,347。

| 模型 | RMSE | MAE | R² |
|---|---|---|---|
| LinearRegression | 0.7647 | 0.5662 | 0.4495 |
| GBTRegressor | 0.7037 | 0.5168 | 0.5338 |

人气是最强解释变量（`corr(rating, log_members) = 0.648`，标准化系数 +0.572），题材数量次之
（+0.124），篇幅几乎无关（+0.061）；但篇幅在 GBT 特征重要度中排第二（0.121），说明它与评分
并非线性关系——单集短片和超长篇评分偏低，12~24 集的季番评分最高。

## 6. 不足与改进

- 受 5.9 GB 物理内存限制，executor 只能给到 900 MB × 2，ALS 只用了 16% 的有效评分训练，rank 上限止步 20。
- 隐因子模型对冷门条目预测分偏高且无界（原始 Top-5 出现过 10.80 分），生成推荐列表时加 `members ≥ 2000` 的人气下界过滤。
- 冷启动问题未解决，`coldStartStrategy='drop'` 会直接丢弃新用户/新番剧。
- 离线批处理，未接入 Kafka / Spark Streaming 做实时推荐。
- 评估指标只有 RMSE / MAE，缺少 Precision@K、NDCG 等排序指标。
