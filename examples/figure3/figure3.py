#!/usr/bin/env python3

# python /mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/figure3/figure3.py \
#   --sw_cdf /mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-11-11/17-32-46/data/joint_gradient_descent_traces.npz \
#   --sw_icdf /mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-10-22/19-29-34/data/joint_gradient_descent_traces.npz \
#   --sw_dsort /mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-11-11/17-43-59/data/joint_gradient_descent_traces.npz \
#   --l2 /mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-11-04/18-06-30/data/joint_gradient_descent_traces.npz \
#   --output /mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/figure3/figure3e.png

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

GROUND_TRUTH = {
    "cvs": 0.0,
    "defoci": 5000.0,
    "offset_xs": 0.0,
    "offset_ys": 0.0,
    "noise_backgrounds": 1.0,
}

ZERO_MIN_BASED_PARAMETERS = {
    "noise_backgrounds",
}

ZERO_MAX_BASED_PARAMETERS = {
    "offset_xs",
    "offset_ys",
}

ST_COLORS = dict(
    zip(
        ["sw_cdf", "sw_dsort", "sw_icdf", ],
        sns.color_palette("Oranges_d", n_colors=3),
    )
)

PARAMETERS = [
    "losses",
    "cvs",
    "defoci",
    "geodesic_angles",
    "offset_xs",
    "offset_ys",
    "noise_backgrounds"
]

METHOD_LABELS = {
    "sw_cdf": "CDF",
    "sw_icdf": "iCDF",
    "sw_dsort": "GM",
}

Y_LABELS = {
    "losses": "Normalized\nLoss",
    "cvs": r"$x$",
    "defoci": r"$f$",
    "geodesic_angles": r"$d(R, R^*)$",
    "offset_xs": r"$t_x$",
    "offset_ys": r"$t_y$",
    "noise_backgrounds": r"$b$",
}

ST_LINESTYLES = {
    "sw_cdf": "-",
    "sw_dsort": "-",
    "sw_icdf": "-",
}

def load_traces(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def get_parameter(data, parameter):
    if parameter not in data:
        raise KeyError(
            f"Missing '{parameter}'. Available fields: {list(data.keys())}"
        )

    values = np.asarray(data[parameter]).squeeze()

    if values.ndim != 1:
        raise ValueError(
            f"'{parameter}' must be one-dimensional after squeezing; "
            f"got shape {values.shape}"
        )

    return values


def normalized_iteration(values):
    if len(values) == 1:
        return np.array([0.0])
    return np.linspace(0.0, 1.0, len(values))

def normalize_loss(values):
    return (values - min(values)) / (max(values)  - min(values) )

def main():
    parser = argparse.ArgumentParser(
        description="Create the 5-by-2 joint-inference trace figure."
    )
    parser.add_argument("--sw_cdf", type=Path, required=True)
    parser.add_argument("--sw_icdf", type=Path, required=True)
    parser.add_argument("--sw_dsort", type=Path, required=True)
    parser.add_argument("--l2", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figure3.png"),
        help="Output figure path (default: figure3.png)",
    )
    args = parser.parse_args()

    first_column = {
        "sw_cdf": load_traces(args.sw_cdf),
        "sw_icdf": load_traces(args.sw_icdf),
        "sw_dsort": load_traces(args.sw_dsort),
    }
    mse = load_traces(args.l2)

    fig, axes = plt.subplots(
        nrows=len(PARAMETERS),
        ncols=2,
        figsize=(12, 8),
        constrained_layout=True,
        sharex=False,
    )

    for column in range(2):
        reference_axis = axes[1, column]
        for row in range(2, len(PARAMETERS)):
            axes[row, column].sharex(reference_axis)

    for row in range(1, len(PARAMETERS) - 1):
        axes[row, 1].sharey(axes[row, 0])

    for row, parameter in enumerate(PARAMETERS):
        ax_left, ax_right = axes[row]
        ax_left.set_ylabel(Y_LABELS[parameter])

        for method, traces in first_column.items():
            y = get_parameter(traces, parameter)

            if parameter == "losses":
                y = normalize_loss(y)

            ax_left.plot(
                y,
                color=ST_COLORS[method],
                linestyle=ST_LINESTYLES[method],
                linewidth=1.8,
                label=METHOD_LABELS[method],
            )
            zero_with_buffer = 0.05 * (max(y) - min(y))
            if parameter in ZERO_MIN_BASED_PARAMETERS:
                ax_left.set_ylim(bottom=(min(min(y), -zero_with_buffer)))
            if parameter in ZERO_MAX_BASED_PARAMETERS:
                ax_left.set_ylim(top=(max(max(y), zero_with_buffer)))
 

        if parameter != "noise_backgrounds":
            y_mse = get_parameter(mse, parameter)
            if parameter == "losses":
                y_mse = normalize_loss(y_mse)

            ax_right.plot(
                y_mse,
                color="tab:blue",
                linestyle="-",
                linewidth=1.8,
                label="MSE",
            )
        else:
            ax_right.set_visible(False)

        ax_left.grid(alpha=0.25)
        ax_right.grid(alpha=0.25)

        if parameter in GROUND_TRUTH:
            ground_truth = GROUND_TRUTH[parameter]

            ax_left.axhline(
                ground_truth,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                label="Ground truth",
            )

            if parameter != "noise_backgrounds":
                ax_right.axhline(
                    ground_truth,
                    color="gray",
                    linestyle="--",
                    linewidth=1.5,
                    label="Ground truth",
                )



    axes[0, 0].set_title("ST")
    axes[0, 1].set_title("MSE")
    axes[0, 0].legend()
    axes[0, 1].legend()

    axes[-1, 0].set_xlabel("Iteration")
    axes[-1, 1].set_xlabel("Iteration")

    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()