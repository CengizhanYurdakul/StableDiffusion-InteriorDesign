import cv2
import torch
import argparse

from src.Processor import Processor

parser = argparse.ArgumentParser()
parser.add_argument("--stableDiffusionModelName", default="sd1.5", help="Specifies the model to be used as a Stable Diffusion")
parser.add_argument("--controlnetMethod", default="depth", help="[canny - segmentation - depth - hed - mlsd] - Specifies which method to use in the ControlNet model")
parser.add_argument("--dtype", default="bf16", help="Specifies which tensor type the models should be initialized")
parser.add_argument("--host", default="0.0.0.0", help="Bind socket to this host.  [default:0.0.0.0]")
parser.add_argument("--port", default=8000, help="Bind socket to this port. If 0, an available port will be picked.")
parser.add_argument("--checkInput", default=False, help="A mechanism that checks whether the user input is a room or not using the YOLO detection model")
args = parser.parse_args()

processor = Processor(args)

promptDict = {
    "scandinavian": "scandinavian forest cabin, log walls, reindeer hide rugs, northern wilderness escape",
    "hollywood": "Hollywood regency glamor, mirrored furniture, velvet upholstery, crystal chandeliers, classic elegance",
    "victorian": "Victorian gothic, dark oak bookshelves, leather chesterfield sofas, ornate chandeliers, moody atmosphere",
    "urban": "urban industrial room, exposed brick walls, polished concrete floors, minimalist decor, edison bulb lighting",
    "coffeshop": "industrial chic room, coffee shop, repurposed factory equipment, reclaimed wood tables, exposed ductwork",
    "cyberpunk": "cyberpunk underground room, neon lights, holographic art installations, retractable skylight, futuristic hideaway",
    "nouveau": "art nouveau room, curved floral patterns, tiffany stained glass, elegance",
    "egyptian": "Egyptian pharaoh's room, gold leaf accents, hieroglyphic murals, timeless royalty",
    "french": "French provencal room, lavender fields color scheme, rustic farmhouse table, sunflower arrangements, country charm",
    "moroccan": "Moroccan riad room, colorful zellige tiles, mosaic backsplash, open-concept layout, exotic",
    "deco": "art deco penthouse room, sunburst patterns, gilded mirrors, crystal chandeliers, roaring twenties opulence",
    "italian": "Italian room, pastel color scheme, gelato display cases, mosaic tile flooring",
    "retro": "retro 50s room, checkered floors, chrome accents, vinyl booth seating, nostalgic dining experience",
    "cozy": "cozy coastal cottage room, white walls, nautical decor, wicker furniture, seaside retreat",
}

#! Test
import os
import numpy as np
from time import time

inputPath = "Inputs"
outputPath = "Outputs"
dirs = os.listdir(inputPath)


timeList = []
for style in list(promptDict.keys()):
    for imageName in dirs:
        image = cv2.imread(os.path.join(inputPath, imageName))
        s = time()
        outputImage, _ = processor.main(image, style, color="none")
        e = time()
        timeList.append(e-s)
        torch.cuda.empty_cache()
        concat = np.concatenate([cv2.resize(image, (np.array(outputImage).shape[1], np.array(outputImage).shape[0])),  cv2.cvtColor(np.array(outputImage), cv2.COLOR_RGB2BGR)], axis=1)
        cv2.imwrite(os.path.join(outputPath, style + "_" + args.controlnetMethod + "_" + imageName), concat)
        
print("FPS: ", 1/np.mean(timeList))