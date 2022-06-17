from .transform_points import NGLink
from .visualization import create_viz_link_from_json
import numpy as np
from tqdm import tqdm


all_somas_path = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_soma/wholebrain_results/all_somas_8608.txt"

points_ng_json = []
with open(all_somas_path, "r") as f:
    lines = f.readlines()
    for line in tqdm(lines, desc="parsing coordinates"):
        if line != "\n":
            line = " ".join(line.split())
            elements = line.split(",")
            coord = [float(elements[0][1:]), float(elements[1]), float(elements[2][:-1])]
            points_ng_json.append(coord)

        
print(f"{len(points_ng_json)} total somas")


viz_link = "https://viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=NRFI2aWmv3d0Ww"

viz_link = NGLink(viz_link.split("json_url=")[-1])
ngl_json = viz_link._json
ngl_json['layers'].append(
    {
        "type": "annotation",
        "points": points_ng_json,
        "name": "transformed_points"
    }   
)

viz_link = create_viz_link_from_json(ngl_json, neuroglancer_link="https://viz.neurodata.io/?json_url=")
print(viz_link)