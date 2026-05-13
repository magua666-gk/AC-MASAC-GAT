import pickle as pkl
from pathlib import Path

from visualization.plot_formation_curves import generate_formation_curves
from visualization.plot_heatmaps import generate_heatmaps


def test_visualization_helpers_generate_files(tmp_path):
    output_dir = tmp_path / "visualization_outputs"
    output_dir.mkdir(parents=True)

    formation_data = {
        "episodes": [
            {
                "timesteps": [
                    {
                        "step": 0,
                        "leaders": [
                            {
                                "agent_id": 0,
                                "pos_x": 150.0,
                                "pos_y": 420.0,
                                "speed": 12.0,
                                "heading_angle": 0.2,
                            }
                        ],
                        "followers": [
                            {
                                "agent_id": 0,
                                "pos_x": 190.0,
                                "pos_y": 430.0,
                                "speed": 13.0,
                                "heading_angle": 0.3,
                                "leader_distance": 41.2,
                                "formation_distance_error": 8.8,
                            }
                        ],
                    },
                    {
                        "step": 1,
                        "leaders": [
                            {
                                "agent_id": 0,
                                "pos_x": 160.0,
                                "pos_y": 410.0,
                                "speed": 12.5,
                                "heading_angle": 0.25,
                            }
                        ],
                        "followers": [
                            {
                                "agent_id": 0,
                                "pos_x": 200.0,
                                "pos_y": 420.0,
                                "speed": 13.2,
                                "heading_angle": 0.31,
                                "leader_distance": 42.0,
                                "formation_distance_error": 8.0,
                            }
                        ],
                    },
                ]
            }
        ]
    }

    data_path = output_dir / "formation_data.pkl"
    with data_path.open("wb") as f:
        pkl.dump(formation_data, f)

    heatmap_paths = generate_heatmaps(data_path, output_dir)
    curves_path = generate_formation_curves(data_path, output_dir, "AC-MASAC")

    assert heatmap_paths
    for path in heatmap_paths.values():
        assert path.endswith(".png")
        assert Path(path).exists()

    assert curves_path is not None
    assert Path(curves_path).exists()
