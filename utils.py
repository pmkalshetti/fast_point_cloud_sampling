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
    pcd = o3d.io.read_point_cloud(path)
    print(f"Read {np.asarray(pcd.points).shape[0]} points")
    return pcd


def compute_scores_from_points(points, filter_type, scale_min_dist, scale_max_dist):
    scores = graph_filter.compute_scores(points, filter_type, scale_min_dist, scale_max_dist)

    return scores


def sample_points(points_orig, scores, n_samples):
    # normalize scores
    scores = scores / np.sum(scores)

    ids_sampled = np.random.choice(
        points_orig.shape[0], n_samples, replace=False, p=scores)
    points_sampled = points_orig[ids_sampled]
    return points_sampled


def sample_pcd(pcd_orig, filter_type, n_samples, scale_min_dist, scale_max_dist):
    points_orig = np.asarray(pcd_orig.points)
    scores = compute_scores_from_points(points_orig, filter_type, scale_min_dist, scale_max_dist)
    points_sampled = sample_points(points_orig, scores, n_samples)

    pcd_sampled = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_sampled))

    return pcd_sampled


def write_pcd(pcd, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    o3d.io.write_point_cloud(path, pcd, write_ascii=True)


def add_noise(pcd_orig, std):
    """adds normal noise"""
    points_orig = np.asarray(pcd_orig.points)
    points_noisy = points_orig + np.random.normal(scale=std, size=points_orig.shape)
    pcd_noisy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_noisy))

    return pcd_noisy


def create_box(n_points, size):
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    return mesh.sample_points_poisson_disk(n_points)

def create_cone(n_points, radius, height):
    mesh = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height)
    return mesh.sample_points_poisson_disk(n_points)

def create_cylinder(n_points, radius, height):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    return mesh.sample_points_poisson_disk(n_points)

def create_moebius(n_points, radius):
    mesh = o3d.geometry.TriangleMesh.create_moebius(raidus=radius, width=10)
    return mesh.sample_points_poisson_disk(n_points)

def create_sphere(n_points, radius):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    return mesh.sample_points_poisson_disk(n_points)