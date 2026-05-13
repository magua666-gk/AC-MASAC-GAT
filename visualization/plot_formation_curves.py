import os
import pickle as pkl
import time

import matplotlib

matplotlib.use("Agg")

from visualization.matplotlib_fonts import configure_matplotlib_fonts
configure_matplotlib_fonts()
import matplotlib.pyplot as plt
import numpy as np


def _load_formation_data(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def _mean_or_nan(values):
    return float(np.mean(values)) if values else np.nan


def _extract_series(formation_data):
    steps = []
    formation_errors = []
    leader_distances = []
    leader_speeds = []
    follower_speeds = []
    heading_diffs = []

    global_step = 0
    for episode in formation_data.get("episodes", []):
        for timestep in episode.get("timesteps", []):
            leaders = timestep.get("leaders", [])
            followers = timestep.get("followers", [])
            if not leaders and not followers:
                continue

            leader_heading = leaders[0].get("heading_angle") if leaders else None

            steps.append(global_step)
            formation_errors.append(
                _mean_or_nan([float(f["formation_distance_error"]) for f in followers if "formation_distance_error" in f])
            )
            leader_distances.append(
                _mean_or_nan([float(f["leader_distance"]) for f in followers if "leader_distance" in f])
            )
            leader_speeds.append(
                _mean_or_nan([float(leader["speed"]) for leader in leaders if "speed" in leader])
            )
            follower_speeds.append(
                _mean_or_nan([float(f["speed"]) for f in followers if "speed" in f])
            )

            if leader_heading is None:
                heading_diffs.append(np.nan)
            else:
                diffs = []
                for follower in followers:
                    if "heading_angle" in follower:
                        diff = abs(float(follower["heading_angle"]) - float(leader_heading))
                        diffs.append(min(diff, 2 * np.pi - diff))
                heading_diffs.append(_mean_or_nan(diffs))

            global_step += 1

    return {
        "steps": np.asarray(steps, dtype=np.float32),
        "formation_errors": np.asarray(formation_errors, dtype=np.float32),
        "leader_distances": np.asarray(leader_distances, dtype=np.float32),
        "leader_speeds": np.asarray(leader_speeds, dtype=np.float32),
        "follower_speeds": np.asarray(follower_speeds, dtype=np.float32),
        "heading_diffs": np.asarray(heading_diffs, dtype=np.float32),
    }


def _plot_line(ax, x, y, title, ylabel, label=None):
    mask = np.isfinite(y)
    if np.any(mask):
        ax.plot(x[mask], y[mask], linewidth=1.4, label=label)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if label:
        ax.legend()


def generate_formation_curves(formation_pkl_path, output_dir, algorithm_name="AC-MASAC", timestamp=None):
    """Generate a combined formation-analysis curve plot.

    Args:
        formation_pkl_path: Path to formation_data.pkl.
        output_dir: Directory where the png file will be written.
        algorithm_name: Name shown in the plot title.

    Returns:
        Path to the generated image, or None if no timestep data exists.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    formation_data = _load_formation_data(formation_pkl_path)
    series = _extract_series(formation_data)

    steps = series["steps"]
    if steps.size == 0:
        return None

    output_path = os.path.join(output_dir, f"formation_curves_{timestamp}.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{algorithm_name} Formation Analysis", fontsize=14)

    _plot_line(
        axes[0, 0],
        steps,
        series["formation_errors"],
        "Formation Distance Error",
        "Error",
    )
    _plot_line(
        axes[0, 1],
        steps,
        series["leader_distances"],
        "Leader-Follower Distance",
        "Distance",
    )

    _plot_line(
        axes[1, 0],
        steps,
        series["leader_speeds"],
        "Speed",
        "Speed",
        label="Leader",
    )
    _plot_line(
        axes[1, 0],
        steps,
        series["follower_speeds"],
        "Speed",
        "Speed",
        label="Followers mean",
    )

    _plot_line(
        axes[1, 1],
        steps,
        series["heading_diffs"],
        "Heading Difference",
        "Radians",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
