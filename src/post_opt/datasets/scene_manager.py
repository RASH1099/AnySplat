# COLMAP - Structure-from-Motion and Multi-View Stereo
# Copyright (c) 2018 ETH Zurich, Photogrammetry and Remote Sensing Group
#
# Author: Johannes L. Schönberger <jsch at inf.ethz.ch>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

import os
import sys
import struct
import numpy as np
from collections import defaultdict

BIN_FILE_HEADER = "# bin file"
TXT_FILE_HEADER = "# txt file"

CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
    "UNIFIED": 11,
    "DUAL": 12,
    "SIMPLE_UNIFIED": 13,
    "DOUBLE_SPHERE": 14
}

CAMERA_MODEL_PARAMS_NUM = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
    "UNIFIED": 5,
    "DUAL": 8,
    "SIMPLE_UNIFIED": 4,
    "DOUBLE_SPHERE": 8
}

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = np.array(tuple(map(float, elems[4:])))
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
    return cameras

def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id = read_next_bytes(fid, 8, "Q")[0]
            model_id = read_next_bytes(fid, 4, "i")[0]
            width = read_next_bytes(fid, 8, "Q")[0]
            height = read_next_bytes(fid, 8, "Q")[0]
            model = list(CAMERA_MODEL_IDS.keys())[list(CAMERA_MODEL_IDS.values()).index(model_id)]
            num_params = CAMERA_MODEL_PARAMS_NUM[model]
            params = np.array(read_next_bytes(fid, 8 * num_params, "d" * num_params))
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
    return cameras


def read_images_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue
            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            camera_id = int(elems[8])
            name = elems[9]
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
            line = fid.readline().strip()
            elems = line.split()
    return images


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 8, "Q")[0]
            qvec = np.array(read_next_bytes(fid, 8 * 4, "d" * 4))
            tvec = np.array(read_next_bytes(fid, 8 * 3, "d" * 3))
            camera_id = read_next_bytes(fid, 8, "Q")[0]
            name_length = read_next_bytes(fid, 4, "I")[0]
            name = fid.read(name_length).decode("utf-8")
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
    return images
def read_points3D_text(path):
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue
            elems = line.split()
            point3D_id = int(elems[0])
            xyz = np.array(tuple(map(float, elems[1:4])))
            rgb = np.array(tuple(map(int, elems[4:7])))
            error = float(elems[7])
            track_elems = elems[8:]
            track = np.array(list(map(int, track_elems)))
            points3D[point3D_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track
            }
    return points3D


def read_points3D_binary(path):
    points3D = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            point3D_id = read_next_bytes(fid, 8, "Q")[0]
            xyz = np.array(read_next_bytes(fid, 8 * 3, "d" * 3))
            rgb = np.array(read_next_bytes(fid, 8 * 3, "d" * 3)).astype(int)
            error = read_next_bytes(fid, 8, "d")[0]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, 8 * track_length, "Q" * track_length)
            points3D[point3D_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": np.array(track_elems)
            }
    return points3D


def read_cameras(model_path):
    if model_path.endswith(".txt"):
        return read_cameras_text(model_path)
    elif model_path.endswith(".bin"):
        return read_cameras_binary(model_path)
    else:
        raise Exception("Unknown camera file format")


def read_images(model_path):
    if model_path.endswith(".txt"):
        return read_images_text(model_path)
    elif model_path.endswith(".bin"):
        return read_images_binary(model_path)
    else:
        raise Exception("Unknown image file format")


def read_points3D(model_path):
    if model_path.endswith(".txt"):
        return read_points3D_text(model_path)
    elif model_path.endswith(".bin"):
        return read_points3D_binary(model_path)
    else:
        raise Exception("Unknown points3D file format")


class SceneManager(object):
    def __init__(self, database_path, model_path):
        self.database_path = database_path
        self.model_path = model_path
        self.images = {}
        self.cameras = {}
        self.points3D = {}
        self.db_con = None
        self.db_cur = None

    def load_cameras(self):
        cameras_path_bin = os.path.join(self.model_path, "cameras.bin")
        cameras_path_txt = os.path.join(self.model_path, "cameras.txt")
        if os.path.exists(cameras_path_bin):
            self.cameras = read_cameras(cameras_path_bin)
        elif os.path.exists(cameras_path_txt):
            self.cameras = read_cameras(cameras_path_txt)
        else:
            raise IOError("cameras.{txt,bin} not found")

    def load_images(self):
        images_path_bin = os.path.join(self.model_path, "images.bin")
        images_path_txt = os.path.join(self.model_path, "images.txt")
        if os.path.exists(images_path_bin):
            self.images = read_images(images_path_bin)
        elif os.path.exists(images_path_txt):
            self.images = read_images(images_path_txt)
        else:
            raise IOError("images.{txt,bin} not found")

    def load_points3D(self):
        points_path_bin = os.path.join(self.model_path, "points3D.bin")
        points_path_txt = os.path.join(self.model_path, "points3D.txt")
        if os.path.exists(points_path_bin):
            self.points3D = read_points3D(points_path_bin)
        elif os.path.exists(points_path_txt):
            self.points3D = read_points3D(points_path_txt)
        else:
            raise IOError("points3D.{txt,bin} not found")

    def load(self):
        self.load_cameras()
        self.load_images()
        self.load_points3D()

    ######################################################################
    # DATABASE ACCESS
    ######################################################################

    def open_database(self):
        if not os.path.exists(self.database_path):
            raise IOError(f"Database path does not exist: {self.database_path}")
        import sqlite3
        self.db_con = sqlite3.connect(self.database_path)
        self.db_cur = self.db_con.cursor()

    def close_database(self):
        if self.db_con is not None:
            self.db_con.close()
            self.db_con = None
            self.db_cur = None

    def __enter__(self):
        self.open_database()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close_database()

    def execute(self, sql, args=None):
        if self.db_cur is None:
            raise RuntimeError("Database not opened")
        if args is None:
            self.db_cur.execute(sql)
        else:
            self.db_cur.execute(sql, args)
        return self.db_cur

    def fetchone(self):
        return self.db_cur.fetchone()

    def fetchall(self):
        return self.db_cur.fetchall()

    def commit(self):
        self.db_con.commit()

    ######################################################################
    # MATCHES & GEOMETRY
    ######################################################################

    def pair_id_to_image_ids(self, pair_id):
        image_id1 = pair_id >> 32
        image_id2 = pair_id & 0xFFFFFFFF
        return image_id1, image_id2

    def image_ids_to_pair_id(self, image_id1, image_id2):
        if image_id1 > image_id2:
            image_id1, image_id2 = image_id2, image_id1
        return (image_id1 << 32) | image_id2

    def get_matches(self, image_id1, image_id2):
        pair_id = self.image_ids_to_pair_id(image_id1, image_id2)
        sql = "SELECT data FROM matches WHERE pair_id=?"
        rows = self.execute(sql, (pair_id,)).fetchall()
        if len(rows) == 0:
            return None
        data = np.fromstring(rows[0][0], dtype=np.uint32)
        return data.reshape(-1, 2)

    def get_two_view_geometry(self, image_id1, image_id2):
        pair_id = self.image_ids_to_pair_id(image_id1, image_id2)
        sql = "SELECT data FROM two_view_geometries WHERE pair_id=?"
        rows = self.execute(sql, (pair_id,)).fetchall()
        if len(rows) == 0:
            return None

        data = np.fromstring(rows[0][0], dtype=np.float64)
        return data

    ######################################################################
    # KEYPOINTS (FEATURES)
    ######################################################################

    def get_keypoints(self, image_id):
        sql = "SELECT data FROM keypoints WHERE image_id=?"
        rows = self.execute(sql, (image_id,)).fetchall()
        if len(rows) == 0:
            return None
        data = np.fromstring(rows[0][0], dtype=np.float32)
        return data.reshape(-1, 2)

    def get_descriptors(self, image_id):
        sql = "SELECT data FROM descriptors WHERE image_id=?"
        rows = self.execute(sql, (image_id,)).fetchall()
        if len(rows) == 0:
            return None
        data = np.fromstring(rows[0][0], dtype=np.uint8)
        return data.reshape(-1, 128)

    def get_num_keypoints(self, image_id):
        sql = "SELECT rows FROM keypoints WHERE image_id=?"
        rows = self.execute(sql, (image_id,)).fetchall()
        if len(rows) == 0:
            return 0
        return rows[0][0]

    ######################################################################
    # IMAGE NAME MAP
    ######################################################################

    def image_name_to_id_map(self):
        """Return mapping from image name to image id."""
        sql = "SELECT image_id, name FROM images"
        rows = self.execute(sql).fetchall()
        return {row[1]: row[0] for row in rows}

    def image_id_to_name_map(self):
        """Return mapping from image id to image name."""
        sql = "SELECT image_id, name FROM images"
        rows = self.execute(sql).fetchall()
        return {row[0]: row[1] for row in rows}

    ######################################################################
    # IMAGE-PAIR UTILITIES
    ######################################################################

    def get_image_pairs(self):
        """List all image pairs in database."""
        sql = "SELECT pair_id FROM two_view_geometries"
        rows = self.execute(sql).fetchall()
        return [row[0] for row in rows]

    def get_image_pairs_ids(self):
        """Return image id pairs as tuples."""
        pairs = self.get_image_pairs()
        return [self.pair_id_to_image_ids(p) for p in pairs]

    ######################################################################
    # READ / WRITE .bin and .txt MODELS
    ######################################################################

def write_cameras_text(cameras, path):
    with open(path, "w") as fid:
        fid.write("# Camera list with one line per camera:\n")
        fid.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for camera_id, camera in cameras.items():
            params_str = " ".join(map(str, camera["params"]))
            fid.write(
                f"{camera_id} {camera['model']} {camera['width']} {camera['height']} {params_str}\n"
            )

def write_images_text(images, path):
    with open(path, "w") as fid:
        fid.write("# Image list with one line per image:\n")
        fid.write("#   IMAGE_ID, QW QX QY QZ, TX TY TZ, CAMERA_ID, NAME\n")
        for image_id, img in images.items():
            qvec_str = " ".join(map(str, img["qvec"]))
            tvec_str = " ".join(map(str, img["tvec"]))
            fid.write(
                f"{image_id} {qvec_str} {tvec_str} {img['camera_id']} {img['name']}\n"
            )
        fid.write("\n")

def write_points3D_text(points3D, path):
    with open(path, "w") as fid:
        fid.write("# 3D point list with one line per point:\n")
        fid.write("#   POINT3D_ID, XYZ, RGB, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for point3D_id, p in points3D.items():
            xyz_str = " ".join(map(str, p["xyz"]))
            rgb_str = " ".join(map(str, p["rgb"]))
            track_str = " ".join(map(str, p["track"]))
            fid.write(
                f"{point3D_id} {xyz_str} {rgb_str} {p['error']} {track_str}\n"
            )

def write_model(cameras, images, points3D, path, ext=".txt"):
    os.makedirs(path, exist_ok=True)
    write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
    write_images_text(images, os.path.join(path, "images" + ext))
    write_points3D_text(points3D, os.path.join(path, "points3D" + ext))


###########################################################################
# BINARY READERS FOR .bin MODEL FILES
###########################################################################

def read_model(path):
    cameras = read_cameras(os.path.join(path, "cameras.bin"))
    images  = read_images(os.path.join(path, "images.bin"))
    points3D = read_points3D(os.path.join(path, "points3D.bin"))
    return cameras, images, points3D


###########################################################################
# CONVERSION UTILITIES
###########################################################################

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
        ],
        [
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
        ],
        [
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
    ])


def rotmat2qvec(R):
    """Convert rotation matrix to quaternion."""
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], R[1, 0] + R[0, 1], R[2, 0] + R[0, 2], R[1, 2] - R[2, 1]],
        [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], R[2, 1] + R[1, 2], R[2, 0] - R[0, 2]],
        [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], R[0, 1] - R[1, 0]],
        [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]],
    ])
    K /= 3.0
    w, x, y, z = np.linalg.eigh(K)[1][:, 3]
    return np.array([w, x, y, z])

def invert_pose(qvec, tvec):
    """Invert camera pose."""
    R = qvec2rotmat(qvec)
    R_inv = R.T
    t_inv = -R_inv @ tvec
    q_inv = rotmat2qvec(R_inv)
    return q_inv, t_inv
def compute_camera_center(qvec, tvec):
    """Compute camera center in world coordinates."""
    R = qvec2rotmat(qvec)
    return -R.T @ tvec


def read_model_ext(path):
    cameras = read_cameras(os.path.join(path, "cameras.txt"))
    images = read_images(os.path.join(path, "images.txt"))
    points3D = read_points3D(os.path.join(path, "points3D.txt"))
    return cameras, images, points3D


def write_model_ext(cameras, images, points3D, path):
    os.makedirs(path, exist_ok=True)
    write_cameras_text(cameras, os.path.join(path, "cameras.txt"))
    write_images_text(images, os.path.join(path, "images.txt"))
    write_points3D_text(points3D, os.path.join(path, "points3D.txt"))


def append_to_model(cameras, images, points3D, append_cameras, append_images, append_points3D):
    """Merge two COLMAP models into the first set."""
    for cam_id, cam in append_cameras.items():
        if cam_id not in cameras:
            cameras[cam_id] = cam

    for img_id, img in append_images.items():
        if img_id not in images:
            images[img_id] = img

    for pid, p in append_points3D.items():
        if pid not in points3D:
            points3D[pid] = p

    return cameras, images, points3D


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="COLMAP scene manager")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--database_path", required=False)
    parser.add_argument("--output_path", required=False)

    args = parser.parse_args()

    print("Loading model from:", args.model_path)
    cameras, images, points3D = read_model(args.model_path)

    print("Loaded cameras:", len(cameras))
    print("Loaded images:", len(images))
    print("Loaded points3D:", len(points3D))

    if args.output_path:
        print("Writing model to:", args.output_path)
        write_model(cameras, images, points3D, args.output_path)
