import numpy as np
import open3d as o3d
import argparse
from utils import sample_pcd, draw, write_pcd, read_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_ply", help="path to ply file")
    parser.add_argument("-n", "--n_samples", default=1000, type=int, help="number of samples")
    parser.add_argument("-s", "--save", default="", help="path to save output ply file")
    parser.add_argument("-f", "--filter_type", default="high", help="filter type to use")
    args = parser.parse_args()

    pcd_orig = read_data(args.path_ply)
    pcd_orig.paint_uniform_color([0, 0, 0])  # original points are blacks

    pcd_sampled = sample_pcd(pcd_orig, args.filter_type, args.n_samples, 10, 10)
    pcd_sampled.paint_uniform_color([1, 0, 0])  # sampled points are red

    if args.save:
        write_pcd(pcd_sampled, args.save)
    
    draw([pcd_orig.translate([-20, 0, 0]), pcd_sampled])

