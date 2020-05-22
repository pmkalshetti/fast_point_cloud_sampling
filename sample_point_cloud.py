import numpy as np
import open3d as o3d
import argparse


def read_data(path):
    return o3d.io.read_point_cloud(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_ply", help="path to ply file")
    args = parser.parse_args()

    pcd_orig = read_data(args.path_ply)
    pcd_orig.paint_uniform_color([0, 0, 0])
    points_orig = np.asarray(pcd_orig.points)