# local imports
from .util import aws_cli
from .visualization import create_viz_link_from_json, get_neuroglancer_json

import pathlib
import subprocess
import shlex
import requests as r
import numpy as np
import h5py
from cloudvolume import CloudVolume
from collections import defaultdict
import uuid
import argparse
from scipy.io import loadmat
import json
import random


def loadmat_v73(mat_path):
    arrays = {}
    f = h5py.File(mat_path, "r")
    for k, v in f.items():
        arrays[k] = np.array(v)
    return arrays


class NGLink:
    def __init__(self, json_link):
        self.points = defaultdict(lambda: "")
        self.json_link = json_link
        self._set_json_from_link()

    def get_annotations(self, points, desc=True):
        annotations = []
        for i, j in points.items():
            x = {
                "point": j.tolist(),
                "type": "point",
                "id": f"{uuid.uuid1().hex}",
            }
            if desc: # because I would get errors when this was a simple integer or string or something
                x["description"] = i
            annotations.append(x)
        return annotations

    def get_points_in(self, coordinate_system):
        if coordinate_system == "voxel":
            return self.points
        else:
            return {i[0]: (i[1] * self.points_voxel_size) for i in self.points.items()}

    def _set_json_from_link(self):
        self._json = r.get(self.json_link).json()
        self._parse_voxel_size()
        self.output_dim = [self._json["dimensions"][i] for i in self._json["dimensions"].keys()]
        self.layers = [self._parse_layer(i) for i in self._json["layers"]]

    def _parse_layer(self, layer_data):
        if layer_data["type"] == "image":
            return self._parse_image_layer(layer_data)
        elif layer_data["type"] == "annotation":
            return self._parse_annotation_layer(layer_data)
        else:
            return

    def _parse_annotation_layer(self, layer_data):
        # points in physical units
        for i in layer_data["annotations"]:
            if i["type"] != "point":
                continue
            if "description" in i.keys():
                self.points[i["description"].strip()] = i["point"]
            else:
                self.points[f"{i['id']}"] = i["point"]
        return layer_data

    def _parse_image_layer(self, layer_data):
        if isinstance(layer_data["source"], str):
            vol = CloudVolume(layer_data["source"])
        else:
            path = layer_data["source"]
            if not isinstance(path, str):
                path = path["url"]
            vol = CloudVolume(path.split("precomputed://")[-1])
        self.image_shape = np.array(vol.scales[0]["size"])
        # converting from nm to um
        self.image_voxel_size = np.array(vol.scales[0]["resolution"]) / 1e3
        self.voxel_origin = self.image_shape / 2
        self.physical_origin = self.voxel_origin * self.image_voxel_size
        return layer_data

    def _parse_voxel_size(self):
        dims = self._json["dimensions"]
        x_size_m, y_size_m, z_size_m = dims["x"][0], dims["y"][0], dims["z"][0]
        # converting from m to um
        self.points_voxel_size = np.array([x_size_m, y_size_m, z_size_m]) * 1e6


class Fiducial:
    def __init__(self, point, orientation, image_shape, voxel_size, description=""):
        """
        point: 3D point in physical space of fiducial (array-like len 3)
        image_size: size in physical units of native res image in each dim (array-like len 3)
        """
        self.image_shape = np.asarray(image_shape)
        self.voxel_size = np.asarray(voxel_size)
        self._set_origin()
        self.point = np.asarray(point) - self.origin
        self.description = description
        self.orientation = orientation
        self.ng_point = np.asarray(point)

    def _set_origin(self):
        self.origin = (self.image_shape - 1) * self.voxel_size / 2

    def reorient_point(self, out_orient):
        dimension = len(self.point)
        in_orient = str(self.orientation).lower()
        out_orient = str(out_orient).lower()

        inDirection = ""
        outDirection = ""
        orientToDirection = {"r": "r", "l": "r", "s": "s", "i": "s", "a": "a", "p": "a"}
        for i in range(dimension):
            try:
                inDirection += orientToDirection[in_orient[i]]
            except BaseException:
                raise Exception("in_orient '{0}' is invalid.".format(in_orient))

            try:
                outDirection += orientToDirection[out_orient[i]]
            except BaseException:
                raise Exception("out_orient '{0}' is invalid.".format(out_orient))

        if len(set(inDirection)) != dimension:
            raise Exception("in_orient '{0}' is invalid.".format(in_orient))
        if len(set(outDirection)) != dimension:
            raise Exception("out_orient '{0}' is invalid.".format(out_orient))

        order = []
        flip = []
        for i in range(dimension):
            j = inDirection.find(outDirection[i])
            order += [j]
            flip += [in_orient[j] != out_orient[i]]
        new_point = self._flip_point(self.point, axis=flip)
        new_point = new_point[order]
        # update self
        self.point = new_point
        self.orientation = out_orient

        return new_point

    def _reorient_point(self, out_orient):
        dimension = len(self.point)
        in_orient = str(self.orientation).lower()
        out_orient = str(out_orient).lower()

        inDirection = ""
        outDirection = ""
        orientToDirection = {"r": "r", "l": "r", "s": "s", "i": "s", "a": "a", "p": "a"}
        for i in range(dimension):
            try:
                inDirection += orientToDirection[in_orient[i]]
            except BaseException:
                raise Exception("in_orient '{0}' is invalid.".format(in_orient))

            try:
                outDirection += orientToDirection[out_orient[i]]
            except BaseException:
                raise Exception("out_orient '{0}' is invalid.".format(out_orient))

        if len(set(inDirection)) != dimension:
            raise Exception("in_orient '{0}' is invalid.".format(in_orient))
        if len(set(outDirection)) != dimension:
            raise Exception("out_orient '{0}' is invalid.".format(out_orient))

        order = []
        flip = []
        for i in range(dimension):
            j = inDirection.find(outDirection[i])
            order += [j]
            flip += [in_orient[j] != out_orient[i]]
        new_point = self._flip_point(self.point, axis=flip)
        new_point = new_point[order]
        # update self
        self.orientation = out_orient
        self.point = new_point

        return new_point

    def _flip_point(self, point, axis=0):
        tmp_point = point.copy()
        tmp_point[axis] = -point[axis]
        return tmp_point

    def __str__(self):
        return f"{self.description}: [{self.point[0]}, {self.point[1]}, {self.point[2]} ]\norientation: {self.orientation}"


def transform_points(
    target_viz_link,
    atlas_viz_link,
    affine_path,
    velocity_path,
    # voxel size of velocity field
    velocity_field_vsize,
    # transformation direction
    # can be 'atlas' or 'target'
    transformation_direction,
):
    # get json link from viz link
    target_viz = NGLink(target_viz_link.split("json_url=")[-1])
    atlas_viz = NGLink(atlas_viz_link.split("json_url=")[-1])

    # get origin-centered fiducials from viz link
    atlas_fiducials = [
        Fiducial(
            j,
            '',
            atlas_viz.image_shape,
            atlas_viz.image_voxel_size,
            description=i,
        )
        for i, j in atlas_viz.get_points_in("physical").items()
    ]
    target_fiducials = [
        Fiducial(
            j,
            '',
            target_viz.image_shape,
            target_viz.image_voxel_size,
            description=i,
        )
        for i, j in target_viz.get_points_in("physical").items()
    ]
    if transformation_direction == 'target': 
        fiducials = atlas_fiducials
        other_fid = target_viz
        viz = target_viz
    else: 
        fiducials = target_fiducials
        other_fid = atlas_viz
        viz = atlas_viz
    try:
        dest_vox_size = other_fid.points_voxel_size
    except:
        dest_vox_size = other_fid.output_dim


    # run matlab command to get transformed fiducials
    # split into sets of 2000 because matlab can only process limited number of points at a time
    if affine_path != "" and velocity_path != "":
        random.shuffle(fiducials)
        points = [i.point for i in fiducials]
        points_chunks = [points[i:i+2000] for i in range(0, len(points), 2000)]
        points_total = []
        for points in points_chunks[:5]:
            points_string = [", ".join(map(str, i)) for i in points]
            points_string = "; ".join(points_string)
            # velocity field voxel size
            v_size = ", ".join(str(i) for i in velocity_field_vsize)
            # get current file path and set path to transform_points
            base_path = pathlib.Path(__file__).parent.parent.absolute() / 'registration'
            # base_path = os.path.expanduser("~/CloudReg/registration")
            transformed_points_path = "./transformed_points.mat"
            matlab_path = 'matlab'
            matlab_command = f"""
                {matlab_path} -nodisplay -nosplash -nodesktop -r \"addpath(\'{base_path}\');Aname=\'{affine_path}\';vname=\'{velocity_path}\';v_size=[{v_size}];points=[{points_string}];points_t = transform_points(points,Aname,vname,v_size,\'{transformation_direction}\');save(\'./transformed_points.mat\',\'points_t\');exit;\"
            """
            subprocess.run(shlex.split(matlab_command),)

            # transformed_points.m created now
            points_t = loadmat(transformed_points_path)["points_t"]
            points_total.append(points_t)
        points_t = np.concatenate(points_total, axis=0)
        points_ng = {i.description: (j + other_fid.physical_origin)/dest_vox_size for i, j in zip(fiducials, points_t)}
        print(f"fiduc len: {len(fiducials)} points shape: {points_t.shape} points type: {type(points_t)}")
        points_ng_json = viz.get_annotations(points_ng)
        with open('./transformed_points.json', 'w') as fp:
            json.dump(points_ng_json, fp)

        ngl_json = atlas_viz._json
        ngl_json['layers'].append(
            {
                "type": "annotation",
                "annotations": points_ng_json,
                "name": "transformed_points"
            }   
        )
        viz_link = create_viz_link_from_json(ngl_json)
        print(f"VIZ LINK WITH TRANSFORMED POINTS: {viz_link}")

    else:
        raise Exception


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Transform points in Neuroglancer from one space to another given a transformation."
    )
    parser.add_argument(
        "--target_viz_link", help="Neuroglancer viz link to target with fiducials labelled.", type=str
    )
    parser.add_argument(
        "--atlas_viz_link", help="Neuroglancer viz link to atlas (optionally with fiducials labelled if transforming to input data space). Default is link to ARA.", 
        type=str,
        default="https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=ifm4oFKOl10eiA"
    )
    parser.add_argument(
        "--affine_path",
        help="S3 path or local path to matlab transformation files. These will be downloaded to compute the fiducial accuracy",
        type=str,
        default="",
    )
    parser.add_argument(
        "--velocity_path",
        help="S3 path ot local matlab transformation files. These will be downloaded to compute the fiducial accuracy",
        type=str,
        default="",
    )
    parser.add_argument(
        "--velocity_voxel_size",
        help="Voxel size of velocity field in microns",
        nargs="+",
        type=float,
        default=[100.0] * 3,
    )
    parser.add_argument(
        "--transformation_direction", help="viz link to atlas with fiducials labelled",
        type=str,
        default='atlas'
    )
    parser.add_argument(
        "--soma_path", help="path to txt file containing soma coordinates in target space",
        type=str,
        default=None
    )
    # parser.add_argument('-ssh_key_path', help='path to identity file used to ssh into given instance')
    # parser.add_argument('-instance_id', help='EC2 Instance ID of instance to run COLM pipeline on.')
    # parser.add_argument('--instance_type', help='EC2 instance type to run pipeline on. minimum r5d.16xlarge',  type=str, default='r5d.16xlarge')

    args = parser.parse_args()

    if args.affine_path.startswith("s3://") or args.affine_path.startswith("http"):
        # download affine mat to local storage
        aws_cli(shlex.split(f"s3 cp {args.affine_path} ./A.mat"))
        args.affine_path = "./A.mat"
    if args.velocity_path.startswith("s3://") or args.velocity_path.startswith("http"):
        # download velocity mat to local storage
        aws_cli(shlex.split(f"s3 cp {args.velocity_path} ./v.mat"))
        args.velocity_path = "./v.mat"

    # read soma points text file then create target link with them
    if args.soma_path is not None:
        target_viz = NGLink(args.target_viz_link.split("json_url=")[-1])
        ngl_json = target_viz._json

        coords = {}
        counter = 0
        with open(args.soma_path) as f:
            for line in f:
                line = ' '.join(line.split())
                parts = line.split(",")
                coord = np.array([float(parts[0][1:]),float(parts[1]),float(parts[2][:-1])])
                coords[str(counter)] = coord
                counter += 1
        annotations = target_viz.get_annotations(coords, desc=False)
        ngl_json['layers'].append(
            {
                "type": "annotation",
                "annotations": annotations,
                "name": "original_points"
            }   
        )
        target_viz_link = create_viz_link_from_json(ngl_json)
        print(f"VIZ LINK WITH ORIGINAL POINTS: {target_viz_link}")
    else:
        target_viz_link = args.target_viz_link


    transform_points(
        target_viz_link,
        args.atlas_viz_link,
        args.affine_path,
        args.velocity_path,
        args.velocity_voxel_size,
        args.transformation_direction,
    )
