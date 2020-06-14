import argparse
import numpy as np
from utils import sample_pcd, draw, read_data, add_noise, write_pcd


def generate_noiseless(path_data, n_samples, prefix_save, scale_min_dist, scale_max_dist):
    pcd_orig = read_data(path_data)
    pcd_orig.paint_uniform_color([0, 0, 0])  # original points are black

    pcd_all = sample_pcd(pcd_orig, "all", n_samples, scale_min_dist, scale_max_dist)
    write_pcd(pcd_all, f"{prefix_save}all.ply")
    pcd_all.paint_uniform_color([1, 0, 0])

    # pcd_low = sample_pcd(pcd_orig, "low", n_samples, scale_min_dist, scale_max_dist)
    # write_pcd(pcd_low, f"{prefix_save}_low.ply")
    # pcd_low.paint_uniform_color([1, 0, 0])

    pcd_high = sample_pcd(pcd_orig, "high", n_samples, scale_min_dist, scale_max_dist)
    write_pcd(pcd_high, f"{prefix_save}high.ply")
    pcd_high.paint_uniform_color([1, 0, 0])

    translate = 25
    # draw([pcd_orig, pcd_all.translate([translate, 0, 0]), pcd_low.translate([2*translate, 0, 0]), pcd_high.translate([3*translate, 0, 0])])
    draw([pcd_orig, pcd_all.translate([translate, 0, 0]), pcd_high.translate([2*translate, 0, 0])])


def generate_noisy(std, path_data, n_samples, prefix_save, scale_min_dist, scale_max_dist):
    pcd_orig = read_data(path_data)
    pcd_orig = add_noise(pcd_orig, std)
    pcd_orig.paint_uniform_color([0, 0, 0])  # original points are black

    pcd_all = sample_pcd(pcd_orig, "all", n_samples, scale_min_dist, scale_max_dist)
    write_pcd(pcd_all, f"{prefix_save}noisy_all.ply")
    pcd_all.paint_uniform_color([1, 0, 0])

    # pcd_low = sample_pcd(pcd_orig, "low", n_samples, scale_min_dist, scale_max_dist)
    # write_pcd(pcd_low, f"{prefix_save}_noisy_low.ply")
    # pcd_low.paint_uniform_color([1, 0, 0])

    pcd_high = sample_pcd(pcd_orig, "high", n_samples, scale_min_dist, scale_max_dist)
    write_pcd(pcd_high, f"{prefix_save}noisy_high.ply")
    pcd_high.paint_uniform_color([1, 0, 0])

    translate = 25
    # draw([pcd_orig, pcd_all.translate([translate, 0, 0]), pcd_low.translate([2*translate, 0, 0]), pcd_high.translate([3*translate, 0, 0])])
    draw([pcd_orig, pcd_all.translate([translate, 0, 0]), pcd_high.translate([2*translate, 0, 0])])


if __name__ == "__main__":
    np.random.seed(1)
    generate_noiseless("data/box.ply", 1000, "output/box/", 10, 10)
    generate_noiseless("data/cone.ply", 1000, "output/cone/", 10, 10)
    generate_noiseless("data/cylinder.ply", 1000, "output/cylinder/", 10, 10)
    generate_noiseless("data/moebius.ply", 1000, "output/moebius/", 10, 10)
    generate_noiseless("data/sphere.ply", 1000, "output/sphere/", 10, 10)


    # generate_noisy(0.1, "data/cone.ply", 500, "output/cone/", 10, 10)
    # generate_noisy(0.1, "data/box.ply", 500, "output/box/", 5000, 500)