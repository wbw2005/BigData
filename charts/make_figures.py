# -*- coding: utf-8 -*-
"""
make_figures.py —— 把集群跑出来的 CSV 结果画成报告用图（宿主端执行，保证中文不乱码）

输入: final/results/*.csv   （由 pull_results.ps1 从 HDFS 拉回）
输出: final/figures/fig*.png （200 dpi，直接插入 16:9 PPT）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "..", "results")
FIG = os.path.join(BASE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#444444"
plt.rcParams["axes.labelcolor"] = "#222222"
plt.rcParams["xtick.color"] = "#333333"
plt.rcParams["ytick.color"] = "#333333"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#DDDDDD"
plt.rcParams["grid.linewidth"] = 0.8
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["figure.dpi"] = 200

C_MAIN = "#2E5C8A"
C_ACC = "#D1495B"
C_3 = "#4C956C"
C_4 = "#E8A33D"
PALETTE = [C_MAIN, C_ACC, C_3, C_4, "#7B6D8D", "#5AA9E6"]


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        print(f"[skip] {name} 不存在")
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    return df


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] {p}")


def wan(x, _=None):
    return f"{x/10000:.0f}万" if x >= 10000 else f"{x:.0f}"


# ---------------------------------------------------------------- 图1 内容形态
def fig1():
    df = load("sql_q1.csv")
    if df is None:
        return
    df = df.sort_values("anime_cnt", ascending=False)
    fig, ax1 = plt.subplots(figsize=(9.2, 5.0))
    bars = ax1.bar(df["type"], df["anime_cnt"], color=C_MAIN, width=0.62, label="番剧数量")
    ax1.set_ylabel("番剧数量（部）", fontsize=12)
    ax1.set_xlabel("内容形态", fontsize=12)
    ax1.bar_label(bars, fmt="%d", padding=3, fontsize=10, color="#333333")
    ax1.set_ylim(0, df["anime_cnt"].max() * 1.22)

    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(df["type"], df["avg_rating"], "o-", color=C_ACC, lw=2.2, ms=7, label="平均评分")
    for x, y in zip(df["type"], df["avg_rating"]):
        ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=9.5, color=C_ACC)
    ax2.set_ylabel("平均站内评分", fontsize=12, color=C_ACC)
    ax2.tick_params(axis="y", colors=C_ACC)
    ax2.set_ylim(5.0, 7.6)
    ax1.set_title("图1  内容形态分布：TV 占三成，Movie/ONA 评分偏低", fontsize=13.5, pad=12)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=10)
    save(fig, "fig1_type_dist.png")


# ---------------------------------------------------------------- 图2 人气 Top10
def fig2():
    df = load("sql_q2.csv")
    if df is None:
        return
    df = df.sort_values("members")
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars = ax.barh(df["name"], df["members"], color=C_MAIN, height=0.66)
    ax.bar_label(bars, labels=[wan(v) for v in df["members"]], padding=4, fontsize=10)
    ax.set_xlabel("MAL 关注人数（members）", fontsize=12)
    ax.set_title("图2  MAL 人气 Top10：Death Note 以 101 万关注居首", fontsize=13.5, pad=12)
    ax.set_xlim(0, df["members"].max() * 1.16)
    ax.xaxis.set_major_formatter(FuncFormatter(wan))
    ax.tick_params(axis="y", labelsize=10.5)
    save(fig, "fig2_members_top10.png")


# ---------------------------------------------------------------- 图3 口碑 Top10
def fig3():
    df = load("sql_q3.csv")
    if df is None:
        return
    df = df.sort_values("rating")
    labels = [n if len(n) <= 34 else n[:32] + "…" for n in df["name"]]
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    bars = ax.barh(labels, df["rating"], color=C_3, height=0.66)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in df["rating"]], padding=4, fontsize=10)
    ax.set_xlim(8.6, 9.6)
    ax.set_xlabel("站内评分（members ≥ 1 万门槛筛选）", fontsize=12)
    ax.set_title("图3  口碑 Top10：《你的名字。》9.37 分领跑", fontsize=13.5, pad=12)
    for y, (m, c) in enumerate(zip(df["members"], df["user_rating_cnt"])):
        ax.annotate(f"关注 {wan(m)} / 评分 {int(c):,} 人", (8.63, y), va="center",
                    fontsize=8.5, color="#555555")
    ax.tick_params(axis="y", labelsize=10)
    save(fig, "fig3_rating_top10.png")


# ---------------------------------------------------------------- 图4 题材 Top12
def fig4():
    df = load("sql_q4.csv")
    if df is None:
        return
    df = df.sort_values("anime_cnt", ascending=True)
    fig, ax1 = plt.subplots(figsize=(9.2, 5.6))
    bars = ax1.barh(df["genre_name"], df["anime_cnt"], color=C_MAIN, height=0.66)
    ax1.bar_label(bars, fmt="%d", padding=4, fontsize=9.5)
    ax1.set_xlabel("番剧数量（部，一部可多题材）", fontsize=12)
    ax1.set_xlim(0, df["anime_cnt"].max() * 1.18)
    ax1.tick_params(axis="y", labelsize=10.5)

    ax2 = ax1.twiny()
    ax2.grid(False)
    ax2.plot(df["avg_rating"], df["genre_name"], "o", color=C_ACC, ms=7, label="平均评分")
    ax2.set_xlabel("平均站内评分", fontsize=12, color=C_ACC)
    ax2.tick_params(axis="x", colors=C_ACC)
    ax2.set_xlim(5.8, 7.5)
    ax1.set_title("图4  题材 Top12：Comedy 供给最多，Shounen/Drama 口碑更好", fontsize=13.5, pad=12)
    save(fig, "fig4_genre_top12.png")


# ---------------------------------------------------------------- 图5 评分分布
def fig5():
    p = os.path.join(RES, "rating_hist.csv")
    if not os.path.exists(p):
        print("[skip] rating_hist.csv 不存在")
        return
    df = pd.read_csv(p)
    total = df["count"].sum()
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    bars = ax.bar(df["rating"].astype(int), df["count"], color=C_MAIN, width=0.66)
    labels = [f"{c/10000:.1f}万\n{c/total:.1%}" for c in df["count"]]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("用户打分（1~10）", fontsize=12)
    ax.set_ylabel("评分数", fontsize=12)
    ax.set_ylim(0, df["count"].max() * 1.24)
    ax.yaxis.set_major_formatter(FuncFormatter(wan))
    ax.set_title("图5  用户评分分布：7~8 分是绝对主力，整体明显右偏（正向偏差）",
                 fontsize=13.5, pad=12)
    save(fig, "fig5_rating_hist.png")


# ---------------------------------------------------------------- 图6 肘部法
def fig6():
    df = load("kmeans_elbow.csv")
    if df is None:
        return
    fig, ax1 = plt.subplots(figsize=(9.0, 4.9))
    ax1.plot(df["k"], df["wssse"], "o-", color=C_MAIN, lw=2.2, ms=7, label="簇内平方和 WSSSE")
    ax1.set_xlabel("簇数 k", fontsize=12)
    ax1.set_ylabel("WSSSE（越小越紧凑）", fontsize=12, color=C_MAIN)
    ax1.tick_params(axis="y", colors=C_MAIN)
    ax1.set_xticks(df["k"])

    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(df["k"], df["silhouette"], "s--", color=C_ACC, lw=2.0, ms=7, label="轮廓系数")
    ax2.set_ylabel("轮廓系数（越大越好）", fontsize=12, color=C_ACC)
    ax2.tick_params(axis="y", colors=C_ACC)

    ax1.axvline(4, color=C_4, ls=":", lw=2)
    ax1.annotate("选定 k=4", (4, ax1.get_ylim()[1] * 0.95), xytext=(8, 0),
                 textcoords="offset points", color=C_4, fontsize=11, fontweight="bold")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", frameon=False, fontsize=10)
    ax1.set_title("图6  K-Means 肘部法选 k：WSSSE 在 k=4 后边际收益骤降", fontsize=13.5, pad=12)
    save(fig, "fig6_kmeans_elbow.png")


# ---------------------------------------------------------------- 图7 聚类散点
def fig7():
    prof = load("kmeans_profile.csv")
    pts = load("kmeans_points.csv")
    if prof is None or pts is None:
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    lab_map = dict(zip(prof["cluster"], prof["cluster_label"]))
    for i, cl in enumerate(sorted(pts["cluster"].unique())):
        sub = pts[pts["cluster"] == cl]
        ax.scatter(np.log10(sub["members"].clip(lower=1)), sub["rating_f"],
                   s=12, alpha=0.35, color=PALETTE[i % len(PALETTE)],
                   label=f"{lab_map.get(cl, cl)}（{len(sub):,} 部）", linewidths=0)
    # 簇中心
    for i, cl in enumerate(sorted(prof["cluster"])):
        r = prof[prof["cluster"] == cl].iloc[0]
        ax.scatter(np.log10(max(r["avg_members"], 1)), r["avg_rating"], s=260,
                   marker="*", color=PALETTE[i % len(PALETTE)], edgecolors="black", zorder=5)
    ax.set_xlabel("人气 log10(members)", fontsize=12)
    ax.set_ylabel("站内评分", fontsize=12)
    ax.set_title("图7  四类番剧画像：人气与口碑并非同向（★ 为簇中心）", fontsize=13.5, pad=12)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right", markerscale=2.2)
    save(fig, "fig7_kmeans_scatter.png")


# ---------------------------------------------------------------- 图8 ALS 热力图
def fig8():
    df = load("als_grid.csv")
    if df is None:
        return
    piv = df.pivot_table(index="rank", columns="regParam", values="rmse")
    piv = piv.sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:g}" for c in piv.columns], fontsize=11)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"rank={r}" for r in piv.index], fontsize=11)
    ax.set_xlabel("正则化系数 regParam", fontsize=12)
    ax.set_ylabel("隐因子维度 rank", fontsize=12)
    ax.grid(False)
    best = df.loc[df["rmse"].idxmin()]
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            mark = "★" if (piv.index[i] == best["rank"] and abs(piv.columns[j] - best["regParam"]) < 1e-9) else ""
            ax.text(j, i, f"{v:.3f}{mark}", ha="center", va="center", fontsize=11,
                    color="white" if v > np.nanmean(piv.values) else "#222222",
                    fontweight="bold" if mark else "normal")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("测试集 RMSE（越低越好）", fontsize=11)
    ax.set_title(f"图8  ALS 超参网格：最优 rank={int(best['rank'])}, "
                 f"regParam={best['regParam']:g}, RMSE={best['rmse']:.3f}", fontsize=13, pad=12)
    save(fig, "fig8_als_grid.png")


# ---------------------------------------------------------------- 图9 回归
def fig9():
    met = load("reg_metrics.csv")
    coef = load("reg_coef.csv")
    if met is None or coef is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    cn = {"log_members": "log(人气 members)", "episodes_f": "篇幅 episodes",
          "genre_cnt": "题材数量"}

    def nice(x):
        if x in cn:
            return cn[x]
        if x.startswith("type_"):
            return "类型：" + x[5:]
        return x
    coef["label"] = coef["feature"].map(nice)
    coef = coef.sort_values("coef")
    colors = [C_ACC if v < 0 else C_MAIN for v in coef["coef"]]
    bars = ax.barh(coef["label"], coef["coef"], color=colors, height=0.6)
    ax.bar_label(bars, labels=[f"{v:+.3f}" for v in coef["coef"]], padding=4, fontsize=9.5)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_xlabel("标准化回归系数", fontsize=11.5)
    ax.set_xlim(coef["coef"].min() * 1.5 - 0.01, coef["coef"].max() * 1.5 + 0.01)
    ax.set_title("(a) 各因素对评分的边际影响（几乎为 0）", fontsize=12)

    ax = axes[1]
    x = np.arange(len(met))
    w = 0.36
    b1 = ax.bar(x - w / 2, met["r2"], w, color=C_MAIN, label="R²（解释力）")
    b2 = ax.bar(x + w / 2, met["rmse"], w, color=C_4, label="RMSE")
    ax.bar_label(b1, labels=[f"{v:.3f}" for v in met["r2"]], fontsize=10, padding=3)
    ax.bar_label(b2, labels=[f"{v:.3f}" for v in met["rmse"]], fontsize=10, padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(met["model"], fontsize=10.5)
    ax.set_ylim(0, max(met["rmse"].max(), met["r2"].max()) * 1.3)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title("(b) 线性 vs 非线性：预测力都接近 0", fontsize=12)
    fig.suptitle("图9  口碑不可由“篇幅+人气+类型”解释（课程第 12 章 线性回归的应用与边界）",
                 fontsize=13.2, y=1.03)
    save(fig, "fig9_regression.png")


def main():
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
        try:
            f()
        except Exception as e:  # 单张图失败不影响其它
            print(f"[FAIL] {f.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
