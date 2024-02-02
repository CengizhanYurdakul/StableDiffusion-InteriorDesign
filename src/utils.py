import cv2
import torch
import numpy as np
from PIL import Image
from enum import Enum
from io import BytesIO

def processCanny(inputImage:np.array) -> Image:
    """
    It processes the incoming input image with the CannyEdge algorithm.
    It prepares it as output to be given as input to ControlNet.

    Args:
        inputImage (np.array): It is the image in RGB format that the user gives as input.

    Returns:
        Image: Processed image to be used as input to ControlNet.
    """
    image = cv2.Canny(inputImage, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    cannyImage = Image.fromarray(image)
    return cannyImage

def processSegmentation(inputImage:np.array, imageProcessor, imageSegmentator) -> Image:
    """
    It processes the incoming input image with the SemanticSegmentation model.
    It prepares it as output to be given as input to ControlNet.

    Args:
        inputImage (np.array): It is the image in RGB format that the user gives as input.
        imageProcessor (transformers.AutoImageProcessor): Model taken from transformers.AutoImageProcessor
        imageSegmentator (transformers.UperNetForSemanticSegmentation): Model taken from transformers.UperNetForSemanticSegmentation

    Returns:
        Image: Processed image to be used as input to ControlNet.
    """
    inputImage_PIL = Image.fromarray(inputImage)
    pixelValues = imageProcessor(inputImage_PIL, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = imageSegmentator(pixelValues)

    seg = imageProcessor.post_process_semantic_segmentation(outputs, target_sizes=[inputImage_PIL.size[::-1]])[0]
    colorSeg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

    for label, color in enumerate(palette):
        colorSeg[seg == label, :] = color

    colorSeg = colorSeg.astype(np.uint8)
    segmentationImage = Image.fromarray(colorSeg)
    
    return segmentationImage

def processDepth(inputImage:np.array, depthEstimator) -> Image:
    """
    It processes the incoming input image with the DepthEstimation model.
    It prepares it as output to be given as input to ControlNet.

    Args:
        inputImage (np.array): It is the image in RGB format that the user gives as input.
        depthEstimator (transformers.pipeline): Model taken from transformers.pipeline("depth-estimation")

    Returns:
        Image: Processed image to be used as input to ControlNet.
    """
    image = depthEstimator(Image.fromarray(inputImage))['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    depthImage = Image.fromarray(image)
    return depthImage

def processHED(inputImage:np.array, hedDetector) -> Image:
    """
    It processes the incoming input image with the HEDDetector model.
    It prepares it as output to be given as input to ControlNet.

    Args:
        inputImage (np.array): It is the image in RGB format that the user gives as input.
        hedDetector (controlnet_aux.HEDdetector): Model taken from controlnet_aux.HEDdetector.from_pretrained('lllyasviel/ControlNet')

    Returns:
        Image: Processed image to be used as input to ControlNet.
    """
    hedImage = hedDetector(Image.fromarray(inputImage))
    return hedImage

# from controlnet_aux import MLSDdetector
def processMLSD(inputImage:np.array, mlsdDetector) -> Image:
    """
    It processes the incoming input image with the MLSDDetector model.
    It prepares it as output to be given as input to ControlNet.

    Args:
        inputImage (np.array): It is the image in RGB format that the user gives as input.
        mlsdDetector (controlnet_aux.MLSDdetector): Model taken from controlnet_aux.MLSDdetector.from_pretrained('lllyasviel/ControlNet')

    Returns:
        Image: _description_
    """
    mlsdImage = mlsdDetector(Image.fromarray(inputImage))
    return mlsdImage

def readImageFile(data:bytes) -> np.array:
    """
    Converts the image sent by the user from byte format to np.array format.

    Args:
        data (bytes): Image sent by the user

    Returns:
        np.array: Converted image sent by the user
    """
    image = Image.open(BytesIO(data))
    arrayImage = np.array(image)
    return arrayImage

def processImageToReturn(image:Image) -> bytes:
    """
    Converts the output image to be sent to the user from PIL.Image format to byte format.

    Args:
        image (Image): Output image of the Stable Diffusion model

    Returns:
        bytes: Stable Diffusion model output image in byte format to be sent to the user
    """
    imgByteArray = BytesIO()
    image.save(imgByteArray, format="PNG")
    return imgByteArray.getvalue()


# Prompt dictionary with prompts mapped according to the user's desired style. 
# When a new style is to be added, it should be added to this dictionary.
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

# Using Enum-based class for input in FastAPI ensures type safety and restricts user choices
# to predefined options, enhancing code clarity and reducing potential errors.
class Style(str, Enum):
    random = "random"
    scandinavian = "scandinavian"
    hollywood = "hollywood"
    victorian = "victorian"
    urban = "urban"
    coffeshop = "coffeshop"
    cyberpunk = "cyberpunk"
    nouveau = "nouveau"
    egyptian = "egyptian"
    french = "french"
    moroccan = "moroccan"
    deco = "deco"
    italian = "italian"
    retro = "retro"
    cozy = "cozy"

# The palette used for colouring the SemanticSegmentation output according to the classes.
palette = np.asarray([
    [0, 0, 0],
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
])