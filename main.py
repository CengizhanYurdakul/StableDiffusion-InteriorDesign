import cv2
import torch
import argparse

from src.Processor import Processor

parser = argparse.ArgumentParser()
parser.add_argument("--stableDiffusionModelName", default="sd1.5", help="Specifies the model to be used as a Stable Diffusion")
parser.add_argument("--controlnetMethod", default="depth", help="[canny - segmentation - depth - hed - mlsd] - Specifies which method to use in the ControlNet model")
parser.add_argument("--dtype", default="fp16", help="Specifies which tensor type the models should be initialized")
args = parser.parse_args()

processor = Processor(args)

#! Test
import os
import numpy as np

inputPath = "Inputs"
outputPath = "Outputs"
dirs = os.listdir(inputPath)

for imageName in dirs:
    image = cv2.imread(os.path.join(inputPath, imageName))
    outputImage, processedImage = processor.main(image)
    torch.cuda.empty_cache()
    concat = np.concatenate([cv2.resize(image, (np.array(outputImage).shape[1], np.array(outputImage).shape[0])), cv2.resize(np.array(processedImage), (np.array(outputImage).shape[1], np.array(outputImage).shape[0])), cv2.cvtColor(np.array(outputImage), cv2.COLOR_RGB2BGR)], axis=1)
    cv2.imwrite(os.path.join(outputPath, args.controlnetMethod + "_" + imageName), concat)