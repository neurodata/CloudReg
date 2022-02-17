try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse
import requests
from cloudvolume import CloudVolume
import numpy as np
import copy

# 100 um
output_dimensions = [1e-4, 1e-4, 1e-4]

minimum_ngl_json = {
    "dimensions": {
        "x": [output_dimensions[0], "m"],
        "y": [output_dimensions[1], "m"],
        "z": [output_dimensions[2], "m"],
    },
    "layers": [
        {
            "type": "image",
            "source": {
                "url": "",
                "transform": {
                    # last column here is x, y, z translations respectively
                    "matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],],
                    "outputDimensions": {
                        "x": [output_dimensions[0], "m"],
                        "y": [output_dimensions[1], "m"],
                        "z": [output_dimensions[2], "m"],
                    },
                },
            },
            "tab": "source",
            "shader": '#uicontrol vec3 color color(default="white")\n#uicontrol float min slider(default=0, min=0, max=1, step=0.001)\n#uicontrol float max slider(default=1, min=0, max=1, step=0.001)\n#uicontrol float brightness slider(default=0, min=-1, max=1, step=0.1)\n#uicontrol float contrast slider(default=0, min=-3, max=3, step=0.1)\n\nfloat scale(float x) {\n  return (x - min) / (max - min);\n}\n\nvoid main() {\n  emitRGB(\n    color * vec3(\n      scale(\n        toNormalized(getDataValue()))\n       + brightness) * exp(contrast)\n  );\n}',
            "shaderControls": {"max": 0.050},
            "blend": "default",
            "name": "channel",
        },
    ],
    "gpuMemoryLimit": 2000000000,
    "jsonStateServer": "https://json.neurodata.io/v1",
    "layout": "xy",
}

# desired ara resolution in microns
ara_average_data_link = (
    lambda res: f"https://open-neurodata.s3.amazonaws.com/ara_2016/sagittal_{res}um/average_{res}um"
)
ara_annotation_data_link = (
    lambda res: f"https://open-neurodata.s3.amazonaws.com/ara_2016/sagittal_{res}um/annotation_{res}um_2017"
)


def create_viz_link(
    s3_layer_paths,
    affine_matrices=None,
    shader_controls=None,
    url="https://json.neurodata.io/v1",
    neuroglancer_link="https://ara.viz.neurodata.io/?json_url=",
    output_resolution=np.array([1e-4] * 3),
):
    """Create a viz link from S3 layer paths using Neurodata's deployment of Neuroglancer and Neurodata's json state server.

    Args:
        s3_layer_paths (list): List of S3 paths to precomputed volumes to include in the viz link. 
        affine_matrices (list of np.ndarray, optional): List of affine matrices associated with each layer. Affine matrices should be 3x3 for 2D data and 4x4 for 3D data. Defaults to None.
        shader_controls (str, optional): String of shader controls compliant with Neuroglancer shader controls. Defaults to None.
        url (str, optional): URL to JSON state server to store Neueroglancer JSON state. Defaults to "https://json.neurodata.io/v1".
        neuroglancer_link (str, optional): URL for Neuroglancer deployment, default is to use Neurodata deployment of Neuroglancer.. Defaults to "https://ara.viz.neurodata.io/?json_url=".
        output_resolution (np.ndarray, optional): Desired output resolution for all layers in nanometers. Defaults to np.array([1e-4] * 3) nanometers.

    Returns:
        str : viz link to data
    """

    json_data = get_neuroglancer_json(
        s3_layer_paths, affine_matrices, output_resolution
    )
    r = requests.post(url, json=json_data)
    json_url = r.json()["uri"]
    viz_link = f"{neuroglancer_link}{json_url}"
    return viz_link

def create_viz_link_from_json(
    ngl_json,
    url="https://json.neurodata.io/v1",
    neuroglancer_link="https://ara.viz.neurodata.io/?json_url=",
):
    r = requests.post(url, json=ngl_json)
    json_url = r.json()["uri"]
    viz_link = f"{neuroglancer_link}{json_url}"
    return viz_link

def get_neuroglancer_json(s3_layer_paths, affine_matrices, output_resolution):
    """Generate Neuroglancer state json.

    Args:
        s3_layer_paths (list of str): List of S3 paths to precomputed layers.
        affine_matrices (list of np.ndarray): List of affine matrices for each layer.
        output_resolution (np.ndarray): Resolution we want to visualize at for all layers.

    Returns:
        dict: Neuroglancer state JSON
    """
    ngl_json = copy.deepcopy(minimum_ngl_json)
    ngl_json["layers"] = []
    if affine_matrices is None:
        affine_matrices = [None] * len(s3_layer_paths)
    ngl_json["layers"] = [
        get_layer_json(i, j, output_resolution)
        for i, j in zip(s3_layer_paths, affine_matrices)
    ]
    # print(ngl_json['layers'][0]['source'])
    # print(ngl_json['layers'][1]['source'])
    return ngl_json


def get_output_dimensions_json(output_resolution):
    """Convert output dimensions to Neuroglancer JSON

    Args:
        output_resolution (np.ndarray): desired output resolution for precomputed data.

    Returns:
        dict: Neuroglancer JSON for output dimensions
    """
    # output dimensions must be  in meters
    output_json = copy.deepcopy(minimum_ngl_json["dimensions"])
    output_json["x"][0] = output_resolution[0]
    output_json["y"][0] = output_resolution[1]
    output_json["z"][0] = output_resolution[2]
    return output_json

    """
    affine_matrix has translations in microns
    output resolution in meters
    """


def get_layer_json(s3_layer_path, affine_matrix, output_resolution):
    """Generate Neuroglancer JSON for single layer.

    Args:
        s3_layer_path (str): S3 path to precomputed layer.
        affine_matrix (np.ndarray): Affine matrix to apply to current layer. Translation in this matrix is in microns.
        output_resolution (np.ndarray): desired output resolution to visualize layer at.

    Returns:
        dict: Neuroglancer JSON for single layer.
    """
    vol = CloudVolume(s3_layer_path)
    s3_url = S3Url(s3_layer_path)
    # this is in units of m
    # output_resolution = np.array([minimum_ngl_json['dimensions']['x'][0], minimum_ngl_json['dimensions']['y'][0], minimum_ngl_json['dimensions']['z'][0]])

    if affine_matrix is None:
        affine_matrix = np.eye(4)
    else:
        # convert translations from microns to voxels and convert output resolution from m to um
        affine_matrix[:3, -1] /= output_resolution * 1e6
        # mapping from S3 bucket to cloudfront
    mapping = {
        "colm-precomputed-volumes": "https://dlab-colm.neurodata.io",
        "smartspim-precomputed-volumes": "https://dlab-colm.neurodata.io",
        "colm-land-preprocessed": "https://d1o9rpg615hgq7.cloudfront.net"
    }
    
    if s3_url.bucket in list(mapping.keys()):
        url = f"precomputed://{mapping[s3_url.bucket]}/{s3_url.key}"
    else:
        url = f"precomputed://{s3_layer_path}"

    # layer_data['source']['transform']['matrix'] = affine[:3,:].tolist()
    layer_data = {
        "type": vol.layer_type,
        "source": {
            "url": url,
            "transform": {
                # last column here is x, y, z translations respectively
                "matrix": affine_matrix[:3, :].tolist(),
                "outputDimensions": get_output_dimensions_json(output_resolution),
            },
        },
        "tab": "source",
        "shader": '#uicontrol vec3 color color(default="white")\n#uicontrol float min slider(default=0, min=0, max=1, step=0.001)\n#uicontrol float max slider(default=1, min=0, max=1, step=0.001)\n#uicontrol float brightness slider(default=0, min=-1, max=1, step=0.1)\n#uicontrol float contrast slider(default=0, min=-3, max=3, step=0.1)\n\nfloat scale(float x) {\n  return (x - min) / (max - min);\n}\n\nvoid main() {\n  emitRGB(\n    color * vec3(\n      scale(\n        toNormalized(getDataValue()))\n       + brightness) * exp(contrast)\n  );\n}',
        "shaderControls": {"max": 0.05},
        "blend": "default",
        "name": s3_url.key.split("/")[-1],
    }

    return layer_data


# copied because of import issues
# below code from https://stackoverflow.com/questions/42641315/s3-urls-get-bucket-name-and-path
class S3Url(object):
    """
    >>> s = S3Url("s3://bucket/hello/world")
    >>> s.bucket
    'bucket'
    >>> s.key
    'hello/world'
    >>> s.url
    's3://bucket/hello/world'

    >>> s = S3Url("s3://bucket/hello/world?qwe1=3#ddd")
    >>> s.bucket
    'bucket'
    >>> s.key
    'hello/world?qwe1=3#ddd'
    >>> s.url
    's3://bucket/hello/world?qwe1=3#ddd'

    >>> s = S3Url("s3://bucket/hello/world#foo?bar=2")
    >>> s.key
    'hello/world#foo?bar=2'
    >>> s.url
    's3://bucket/hello/world#foo?bar=2'
    """

    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip("/") + "?" + self._parsed.query
        else:
            return self._parsed.path.lstrip("/")

    @property
    def url(self):
        return self._parsed.geturl()
