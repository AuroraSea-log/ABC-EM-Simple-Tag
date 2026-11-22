"""Plot ABC-EM learning curves from saved numeric logs."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    files = sorted((args.output / "runs").glob("seed_*/metrics.csv"))
    if not files:
        parser.error("No completed evaluation logs found")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    for key, ylabel in (("nll", "Negative log-likelihood"), ("kl", "KL divergence"),
                        ("position_coverage_percent", "Pursuer position coverage (%)"),
                        ("joint_position_bins", "Visited joint-position bins")):
        aggregate = frame.groupby("interactions")[key].agg(["mean", "min", "max"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(aggregate.index, aggregate["mean"], color="#137c66", label="ABC-EM")
        ax.fill_between(aggregate.index, aggregate["min"], aggregate["max"], color="#137c66", alpha=.18)
        ax.set(xlabel="Opponent interactions", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"{key}.png", dpi=200)
        fig.savefig(figures / f"{key}.pdf")
        plt.close(fig)
    frame.to_csv(figures / "curve_data.csv", index=False)
    print(figures.resolve())


if __name__ == "__main__":
    main()
