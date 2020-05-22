import numpy as np
import open3d as o3d
import argparse
from build import graph_filter
import os

def get_vis():
    vis = o3d.visualization.Visualizer()
    cwd = os.getcwd()  # to handle issue on OS X
    vis.create_window()
    os.chdir(cwd)  # to handle issue on OS X
    return vis

def draw(geometries):
    vis = get_vis()
    if not isinstance(geometries, list):
        geometries = [geometries]
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.run()
    vis.destroy_window()


def read_data(path):
    return o3d.io.read_point_cloud(path)


def compute_scores_from_points(points, weight_graph=0.8):
    scores = graph_filter.compute_scores(points)

    return scores


def sample_points(points_orig, scores, n_samples):
    # normalize scores
    scores = scores / np.sum(scores)

    ids_sampled = np.random.choice(
        points_orig.shape[0], n_samples, replace=False, p=scores)
    points_sampled = points_orig[ids_sampled]
    return points_sampled



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_ply", help="path to ply file")
    parser.add_argument("-n", "--n_samples", default=1000, type=int, help="number of samples")
    args = parser.parse_args()

    pcd_orig = read_data(args.path_ply)
    pcd_orig.paint_uniform_color([0, 0, 0])  # original points are black

    points_orig = np.asarray(pcd_orig.points)
    scores = compute_scores_from_points(points_orig)
    points_sampled = sample_points(points_orig, scores, args.n_samples)

    pcd_sampled = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(points_sampled))
    pcd_sampled.paint_uniform_color([1, 0, 0])  # sampled points are red

    draw([pcd_orig.translate([-70, 0, 0]), pcd_sampled])
