import torch
import numpy as np

from src.utils import *
from src.Diffusion.DiffusionFactory import StableDiffusionFactory
from src.ControlNet.ControlNetFactory import ControlNetFactory

class Processor:
    def __init__(self, args:str) -> None:
        self.args = args
        
        self.generator = torch.manual_seed(35)
        self.initVariables()
        self.initControlNetModel()
        self.initStableDiffusionModel()
        
        if self.args.controlnetMethod == "depth":
            self.initDepthEstimator()
        elif self.args.controlnetMethod == "segmentation":
            self.initSegmentationNetwork()
        elif self.args.controlnetMethod == "hed":
            
            self.initHEDDetector()
        elif self.args.controlnetMethod == "mlsd":
            
            self.initMLSDDetector()
        
    def initVariables(self):
        self.stableDiffusionModel = None
        self.controlNetModel = None
        self.depthEstimator = None
        self.imageProcessor = None
        self.imageSegmentator = None
        
        if self.args.dtype == "fp16":
            self.args.dtype = torch.float16
            
    def initControlNetModel(self):
        self.controlnetModel = ControlNetFactory(self.args).callModel()
        
    def initStableDiffusionModel(self):
        self.stableDiffusionModel = StableDiffusionFactory(self.args).callModel(self.controlnetModel)
        
    def initDepthEstimator(self):
        from transformers import pipeline
        self.depthEstimator = pipeline("depth-estimation")
    
    def initSegmentationNetwork(self):
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
        self.imageProcessor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        self.imageSegmentator = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
        
    def initHEDDetector(self):
        from controlnet_aux import HEDdetector
        self.hedDetector = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        
    def initMLSDDetector(self):
        from controlnet_aux import MLSDdetector
        self.mlsdDetector = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
        
    def processImage(self):
        if self.args.controlnetMethod == "canny":
            return processCanny(self.inputImage)
        elif self.args.controlnetMethod == "segmentation":
            return processSegmentation(self.inputImage, self.imageProcessor, self.imageSegmentator)
        elif self.args.controlnetMethod == "depth":
            return processDepth(self.inputImage, self.depthEstimator)
        elif self.args.controlnetMethod == "hed":
            return processHED(self.inputImage, self.hedDetector)
        elif self.args.controlnetMethod == "mlsd":
            return processMLSD(self.inputImage, self.mlsdDetector)
        else:
            raise Exception("Sorry, process for %s method not implemented yet!" % self.args.controlnetMethod)
        
    def main(self, inputImage:np.array) -> np.array:
        
        self.inputImage = inputImage
        
        processedImage = self.processImage()
        
        outputImage = self.stableDiffusionModel(
            # "scandinavian style interior design, high resolution, cozy atmosphere, Living room, photorealistic",
            "bohemian style interior design, high resolution, cozy atmosphere, Living room, photorealistic",
            num_inference_steps=50,
            generator=self.generator,
            image=processedImage
            ).images[0]
        
        return outputImage, processedImage
