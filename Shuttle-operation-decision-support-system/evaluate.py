import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

RATIOS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
BASIC = ["TAXI", "STL", "DRV", "DEM", "DEMAND_PER_DRIVER", "TAXI_RATIO", "STL_RATIO",
         "DEM_squared", "STL_squared", "DEMAND_PER_DRIVER_squared", "DEM_x_STL"]
COVERAGE = ["SBS", "SBT", "GMM", "ADS", "VES"]
LOAD = ["TLR", "PCL", "SSR", "SLF_max", "RCT", "RDI", "WSD", "WSHARE"]
SETS = {"basic": BASIC, "+coverage": BASIC + COVERAGE, "+load": BASIC + LOAD,
        "+both": BASIC + COVERAGE + LOAD}


def load(path):
    df = pd.DataFrame([json.loads(line) for line in open(path, encoding="utf-8")])
    df["STL"] = df["k"]
    df["TAXI"] = df["N"] - df["k"]
    df["DRV"] = df["N"]
    df["DEMAND_PER_DRIVER"] = df["DEM"] / df["DRV"]
    df["TAXI_RATIO"] = df["TAXI"] / df["DRV"]
    df["STL_RATIO"] = df["STL"] / df["DRV"]
    df["DEM_squared"] = df["DEM"] ** 2
    df["STL_squared"] = df["STL"] ** 2
    df["DEMAND_PER_DRIVER_squared"] = df["DEMAND_PER_DRIVER"] ** 2
    df["DEM_x_STL"] = df["DEM"] * df["STL"]
    return df


def picks(test, choose):
    return [(g, g.loc[choose(g)]) for _, g in test.groupby("group")]


def dcr(test, choose):
    return np.array([p["CR"] - g["CR"].min() for g, p in picks(test, choose)])


def main():
    train, test = load("data/train.jsonl"), load("data/test.jsonl")

    print(f"{'model':6s} {'features':10s} {'MAE':>7s} {'R2':>6s} {'dCR':>6s} {'exact-k':>7s}")
    for name, feats in SETS.items():
        xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=666)
        test[f"XGB {name}"] = xgb.fit(train[feats], train["CR"]).predict(test[feats])
        test[f"RF-ens {name}"] = np.mean(
            [RandomForestRegressor(n_estimators=200, random_state=s, n_jobs=-1)
             .fit(train[feats], train["CR"]).predict(test[feats]) for s in (1, 42, 666)], axis=0)
    for model in ("XGB", "RF-ens"):
        for name in SETS:
            c = f"{model} {name}"
            exact = np.mean([p["k"] == g.loc[g["CR"].idxmin(), "k"]
                             for g, p in picks(test, lambda g: g[c].idxmin())])
            print(f"{model:6s} {name:10s} {mean_absolute_error(test['CR'], test[c]):.5f} "
                  f"{r2_score(test['CR'], test[c]):.4f} {dcr(test, lambda g: g[c].idxmin()).mean():.4f} "
                  f"{100 * exact:6.1f}%")

    best = int(train.groupby("ratio_idx")["CR"].mean().idxmin())
    fixed = lambda g: g.index[g["ratio_idx"] == best][0]
    nearest = lambda col: lambda g: (g["k"] - g[col].mode()[0]).abs().idxmin()
    print()
    print(f"Best fixed ratio (k/N = {RATIOS[best]:.2f})  dCR {dcr(test, fixed).mean():.4f}")
    print(f"SSE heuristic                  dCR {dcr(test, nearest('SSE_rec_shuttles')).mean():.4f}")
    print(f"DPS heuristic                  dCR {dcr(test, nearest('DPS_rec_shuttles')).mean():.4f}")
    print(f"Random k                       dCR "
          f"{np.mean([(g['CR'] - g['CR'].min()).mean() for _, g in test.groupby('group')]):.4f}")

    print("\nPaired bootstrap, gain in dCR [95% CI]; core / extension cohorts; gain in MAE [95% CI]")
    cohort = test.groupby("group")["cohort"].first().to_numpy()
    rng = np.random.default_rng(11)
    for a, b in [("XGB basic", "XGB +load"), ("XGB basic", "XGB +coverage"),
                 ("XGB +coverage", "XGB +both"), ("XGB basic", "XGB +both"),
                 ("RF-ens basic", "RF-ens +load")]:
        d = dcr(test, lambda g: g[a].idxmin()) - dcr(test, lambda g: g[b].idxmin())
        lo, hi = np.percentile([rng.choice(d, len(d)).mean() for _ in range(5000)], [2.5, 97.5])
        m = ((test[a] - test["CR"]).abs() - (test[b] - test["CR"]).abs()).to_numpy()
        mlo, mhi = np.percentile([rng.choice(m, len(m)).mean() for _ in range(3000)], [2.5, 97.5])
        print(f"{a:13s} -> {b.split()[1]:9s} {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"core {d[cohort == 'core'].mean():+.4f}  ext {d[cohort == 'extension'].mean():+.4f}  "
              f"MAE {m.mean():+.5f} [{mlo:+.5f}, {mhi:+.5f}]")

    print("\nAvoidable cancellations per scenario, (CR(k_hat) - CR(k*)) * |D|")
    for label, choose in [("XGB +load", lambda g: g["XGB +load"].idxmin()),
                          ("XGB basic", lambda g: g["XGB basic"].idxmin()),
                          ("Best fixed ratio", fixed)]:
        ps = picks(test, choose)
        extra = np.mean([(p["CR"] - g["CR"].min()) * p["DEM"] for g, p in ps])
        total = np.mean([p["CR"] * p["DEM"] for _, p in ps])
        print(f"{label:16s} {extra:.1f} of {total:.1f} cancelled")


if __name__ == "__main__":
    main()
