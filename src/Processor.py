import torch
import random
import numpy as np
from loguru import logger
logger.add("logs/file_{time}.log")
logger.level("INITIALIZE", no=38, color="<green>")
logger.level("INFERENCE", no=38, color="<blue>")

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
            
        if self.args.checkInput:
            self.initYOLODetector()
        
    def initVariables(self):
        """
        Defines the variables to be used in the class and makes format changes.
        """
        self.stableDiffusionModel = None
        self.controlNetModel = None
        self.depthEstimator = None
        self.imageProcessor = None
        self.imageSegmentator = None
        self.yoloDetector = None
        
        if self.args.dtype == "fp16":
            self.args.dtype = torch.float16
        
        logger.log("INITIALIZE", "Processor variables initialized!")
            
    def initControlNetModel(self):
        """
        Initialises the ControlNet class. Calls the ControlNet model from Factory 
        according to the method [canny - segmentation - depth - hed - mlsd] to be used.
        """
        self.controlnetModel = ControlNetFactory(self.args).callModel()
        logger.log("INITIALIZE",  "Processor ControlNet initialized!")
        
    def initStableDiffusionModel(self):
        """
        Initialises the Stable Diffusion class. Calls the appropriate Stable Diffusion 
        model according to the ControlNet method to be used.
        """
        self.stableDiffusionModel = StableDiffusionFactory(self.args).callModel(self.controlnetModel)
        logger.log("INITIALIZE",  "Processor Stable Diffusion model initialized!")
        
    def initDepthEstimator(self):
        """
        Initialises the depth estimation structure required for the preprocess.
        """
        from transformers import pipeline
        self.depthEstimator = pipeline("depth-estimation")
        logger.log("INITIALIZE", "Processor Depth estimator initialized!")
    
    def initSegmentationNetwork(self):
        """
        Initialises the semantic segmentation structure required for the preprocess.
        """
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
        self.imageProcessor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        self.imageSegmentator = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
        logger.log("INITIALIZE", "Processor segmentation model initialized!")
        
    def initHEDDetector(self):
        """
        Initialises the HED detector structure required for the preprocess.
        """
        from controlnet_aux import HEDdetector
        self.hedDetector = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        logger.log("INITIALIZE", "Processor HED initialized!")
        
    def initMLSDDetector(self):
        """
        Initialises the MLSD detector structure required for the preprocess.
        """
        from controlnet_aux import MLSDdetector
        self.mlsdDetector = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
        logger.log("INITIALIZE", "Processor MLSD initialized!")
        
    def initYOLODetector(self):
        """
        Initialises the YOLO detector required for input image checking mechanism.
        """
        from ultralytics import YOLO
        self.yoloDetector = YOLO("yolov8n.pt")
        logger.log("INITIALIZE", "Processor YOLO detector initialized!")
    
    def processImage(self) -> Image:
        """
        Processes the input image according to the method to control the output with ControlNet.
        When a new method is added, its preprocess is implemented here.

        Raises:
            Exception: If you try to preprocess a ControlNet method that has not been implemented 
            before, you will end up here.

        Returns:
            Image: Output of the image to be used for ControlNet in PIL.Image format.
        """
        if self.args.controlnetMethod == "canny":
            cannyImage = processCanny(self.inputImage)
            logger.log("INFERENCE", "Image processed with canny!")
            return cannyImage
        elif self.args.controlnetMethod == "segmentation":
            segmentImage = processSegmentation(self.inputImage, self.imageProcessor, self.imageSegmentator)
            logger.log("INFERENCE", "Image processed with segmentator!")
            return segmentImage
        elif self.args.controlnetMethod == "depth":
            depthImage = processDepth(self.inputImage, self.depthEstimator)
            logger.log("INFERENCE", "Image processed with depth estimator!")
            return depthImage
        elif self.args.controlnetMethod == "hed":
            hedImage = processHED(self.inputImage, self.hedDetector)
            logger.log("INFERENCE", "Image processed with HED!")
            return hedImage
        elif self.args.controlnetMethod == "mlsd":
            mlsdImage = processMLSD(self.inputImage, self.mlsdDetector)
            logger.log("INFERENCE", "Image processed with MLSD!")
            return mlsdImage
        else:
            raise Exception("Sorry, process for %s method not implemented yet!" % self.args.controlnetMethod)
        
    def main(self, inputImage:np.array, style:str, color:str) -> Image:
        """
        Processes the input image received from the user. It takes the style that comes as
        input from the prompt dictionary. It takes inference from Stable Diffusion model
        and output image is obtained.

        Args:
            inputImage (np.array): RGB format image received by the user
            style (str): It is the style chosen by the user
            color (str): It is color chosen by the user

        Returns:
            Image: It is the image processed by the Stable Diffusion pipeline. It is returned in Image.PIL format.
        """
        
        self.inputImage = inputImage
        
        # Style to prompt
        if style == "random":
            style = random.choice(list(promptDict.keys()))
            logger.log("INFERENCE", "Random style selected by user given %s" % style)
        else:
            logger.log("INFERENCE", "%s style selected by user" % style)
        
        # Control of household items in input image
        if self.args.checkInput:
            imageFlag = checkInputForRoom(self.inputImage, self.yoloDetector)
            if not imageFlag:
                logger.log("INFERENCE", "No household items found in the photo!")
                return None, {"Error": "No household items found in the photo!"}
        
        # Processing of the input image according to the ControlNet method
        processedImage = self.processImage()
        
        # Color change prompt modification
        prompt = promptDict[style]
        if not color == "none":
            logger.log("INFERENCE", "Color %s selected by user!" % color)
            prompt += ", %s color furnitures" % color
        
        # Stable Diffusion inference with input image and prompt
        outputImage = self.stableDiffusionModel(
            prompt,
            num_inference_steps=50,
            generator=self.generator,
            image=processedImage
            ).images[0]
        logger.log("INFERENCE", "Image generated without any problems.")
        return outputImage, {"Success": "Image generated without any problems."}
