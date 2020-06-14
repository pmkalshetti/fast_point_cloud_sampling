from utils import write_pcd, create_box, create_cone, create_cylinder, create_moebius, create_sphere

if __name__ == "__main__":
    write_pcd(create_box(10000, 10), "data/box.ply")
    write_pcd(create_cone(10000, 10, 30), "data/cone.ply")
    write_pcd(create_cylinder(10000, 10, 30), "data/cylinder.ply")
    write_pcd(create_moebius(10000, 10.0), "data/moebius.ply")
    write_pcd(create_sphere(10000, 10.0), "data/sphere.ply")
    