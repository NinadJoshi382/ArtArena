
''' We perform the following for the preparition for conducting Motif Duels: 
    (1) We prepare simple maps : {"A1" --> "dir to original artwork A1", "A2" --> "dir to original artwork A2",....} 
    (2) With the same key bindings
    (3) With the same key bindings we prepare suffix (defender's style in the compisiton prompt): {"A1" --> "in the style of <artwork A1> by <artist A1>", "A2" --> "in the style of <artwork A2> by <artist A2>",....} 


    motif_json = the json extracted from LLM with the prompt given in section "Motif Extraction" in the Appendix.
    The layout of the motif_json is dict of the form: {"A1" --> [A1's motif 1,A1's motif 2,...],  "A2" --> [A2's motif 1,A2's motif 2,...], ...} 
    
'''
import json 
from typing import Dict, List, Optional, Union 
import glob 
import torch

motif_json = "/workspace/..../Motifs.json" 

with open(motif_json,"r",encoding="utf-8") as f: 
  mapping  = json.load(f) 

suffix_dict = dict()
motif_list = list(mapping.keys())
for k in motif_list: 
  img_name = map_org[k].split("/")[-1].split("__")[-1]
  img_name = img_name.replace(".jpg","")
  artist, artwork = img_name.rsplit("_",1)
  prompt = f"in the style of {artwork} by {artist}."
  suffix_dict[k] = prompt
  print(prompt) 


# "/workspace/.../top20_original/*.jpg" --> directory where the ET_eval saves the original artworks for top candidates. 
t1 = torch.glob.glob("/workspace/.../top20_original/*.jpg")
map_org = dict()
for i in the range(len(t1)):
  map_org[f"A{i+1}"] = t1[i]


