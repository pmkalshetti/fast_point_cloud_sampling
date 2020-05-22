import numpy as np
import open3d as o3d
import argparse
from build import graph_filter


def read_data(path):
    return o3d.io.read_point_cloud(path)


def compute_scores_from_points(points, weight_graph=0.8):
    scores = graph_filter.compute_scores(points)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_ply", help="path to ply file")
    args = parser.parse_args()

    pcd_orig = read_data(args.path_ply)
    pcd_orig.paint_uniform_color([0, 0, 0])

    points_orig = np.asarray(pcd_orig.points)
    scores = compute_scores_from_points(points_orig)
