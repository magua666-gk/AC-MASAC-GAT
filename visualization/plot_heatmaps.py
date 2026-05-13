import os
import pickle as pkl
import time

import matplotlib

matplotlib.use("Agg")

from visualization.matplotlib_fonts import configure_matplotlib_fonts
configure_matplotlib_fonts()
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BOUNDS = (100.0, 600.0, 100.0, 500.0)


def _load_formation_data(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def _iter_timesteps(formation_data):
    for episode in formation_data.get("episodes", []):
        for timestep in episode.get("timesteps", []):
            yield timestep


def _collect_points(formation_data, key):
    points = []
    values = []
    for timestep in _iter_timesteps(formation_data):
        for item in timestep.get(key, []):
            if "pos_x" not in item or "pos_y" not in item:
                continue
            points.append((float(item["pos_x"]), float(item["pos_y"])))
            values.append(float(item.get("formation_distance_error", 0.0)))
    return np.asarray(points, dtype=np.float32), np.asarray(values, dtype=np.float32)


def _plot_count_heatmap(points, title, output_path, bins=50, bounds=DEFAULT_BOUNDS):
    if points.size == 0:
        return None

    xmin, xmax, ymin, ymax = bounds
    heatmap, xedges, yedges = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]],
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(
        heatmap.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(label="Visits")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _plot_error_heatmap(points, values, title, output_path, bins=50, bounds=DEFAULT_BOUNDS):
    if points.size == 0 or values.size == 0:
        return None

    xmin, xmax, ymin, ymax = bounds
    weighted_sum, xedges, yedges = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]],
        weights=values,
    )
    counts, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]],
    )

    avg_error = np.divide(
        weighted_sum,
        counts,
        out=np.zeros_like(weighted_sum, dtype=np.float64),
        where=counts > 0,
    )
    avg_error = np.ma.masked_where(counts == 0, avg_error)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        avg_error.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap="magma",
    )
    plt.colorbar(label="Mean formation error")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def generate_heatmaps(formation_pkl_path, output_dir, timestamp=None):
    """Generate heatmaps from saved formation data.

    Args:
        formation_pkl_path: Path to formation_data.pkl.
        output_dir: Directory where png files will be written.

    Returns:
        Dict mapping plot names to generated image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    formation_data = _load_formation_data(formation_pkl_path)

    leader_points, _ = _collect_points(formation_data, "leaders")
    follower_points, follower_errors = _collect_points(formation_data, "followers")

    generated = {}

    leader_path = _plot_count_heatmap(
        leader_points,
        "Leader Position Heatmap",
        os.path.join(output_dir, f"leader_position_heatmap_{timestamp}.png"),
    )
    if leader_path:
        generated["leader_positions"] = leader_path

    follower_path = _plot_count_heatmap(
        follower_points,
        "Follower Position Heatmap",
        os.path.join(output_dir, f"follower_position_heatmap_{timestamp}.png"),
    )
    if follower_path:
        generated["follower_positions"] = follower_path

    error_path = _plot_error_heatmap(
        follower_points,
        follower_errors,
        "Follower Formation Error Heatmap",
        os.path.join(output_dir, f"formation_error_heatmap_{timestamp}.png"),
    )
    if error_path:
        generated["formation_error"] = error_path

    return generated
