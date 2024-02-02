import warnings
warnings.filterwarnings("ignore")

import random
import argparse
from fastapi import FastAPI, File, UploadFile

from src.Processor import Processor
from src.utils import Style, promptDict, readImageFile

parser = argparse.ArgumentParser()
parser.add_argument("--stableDiffusionModelName", default="sd1.5", help="Specifies the model to be used as a Stable Diffusion")
parser.add_argument("--controlnetMethod", default="depth", help="[canny - segmentation - depth - hed - mlsd] - Specifies which method to use in the ControlNet model")
parser.add_argument("--dtype", default="fp16", help="Specifies which tensor type the models should be initialized")
parser.add_argument("--host", default="0.0.0.0", help="")
parser.add_argument("--port", default=8000, help="")
args = parser.parse_args()

app = FastAPI()

PROCESSOR = Processor(args)

@app.get("/")
def home():
    return {"Hello": "Cengizhan"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...), style: Style = "random"):
    if style == "random":
        style = random.choice(list(promptDict.keys()))
    
    extension = file.filename.split(".")[-1]
    if not (extension in ["jpg", "JPG", "png", "PNG", "jpeg", "JPEG"]):
        return {"Error": "Extension must be [jpg, JPG, png, PNG, jpeg, JPEG"}
    
    pilImage = readImageFile(await file.read())

    return {"Done": "!!!"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=str(args.host), port=int(args.port))