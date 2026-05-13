import pandas as pd

def compare_results(results_path="results/fe_results.csv"):
    df = pd.read_csv(results_path)
    df = df.sort_values("cv_mean", ascending=False).reset_index(drop=True)

    print("\n=== Feature Engineering Comparison ===")
    print(df[["fe_config","cv_mean","cv_std","n_features"]].to_string(index=False))

    best = df.iloc[0]
    baseline = df[df["fe_config"] == "baseline"].iloc[0]
    gain = best["cv_mean"] - baseline["cv_mean"]

    print(f"\nBest config : {best['fe_config']}")
    print(f"F1-macro    : {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
    print(f"Gain vs baseline: +{gain:.4f}")
    return best["fe_config"]

if __name__ == "__main__":
    best_config = compare_results()
    print(f"\nSet FE_CONFIG='{best_config}' trong inference.py")