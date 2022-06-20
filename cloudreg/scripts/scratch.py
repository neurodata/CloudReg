from .transform_points import NGLink
from .visualization import create_viz_link_from_json
import numpy as np
from tqdm import tqdm
import random


all_somas_path = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_soma/wholebrain_results/all_somas_8606.txt"

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
if len(points_ng_json) > 2000:
    random.shuffle(points_ng_json)
    points_ng_json = points_ng_json[:2000]
    name = "detected_somas_partial"
else:
    name = "detected_somas"


viz_link ='https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=SkzvgDNIQyfCTw'


viz_link = NGLink(viz_link.split("json_url=")[-1])
ngl_json = viz_link._json
print(ngl_json)
# ngl_json['layers'].append(
#     {
#         "type": "annotation",
#         "points": points_ng_json,
#         "name": name
#     }   
# )

# viz_link = create_viz_link_from_json(ngl_json, neuroglancer_link="https://viz.neurodata.io/?json_url=")
# print(viz_link)