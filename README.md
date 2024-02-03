# Interior Design App with FastAPI

## Installation

### Available Variables
```
--stableDiffusionModelName [default: sd1.5 | sd1.5]
--controlnetMethod [default: depth | canny - segmentation - depth - hed - mlsd]
--dtype [default: fp16 | none - fp16 - bf16]
--host [default: 0.0.0.0]
--port [default: 8000]
--checkInput [default: False | storeTrue]
```

### Manuel Installation with Docker
```
docker build -t interiordesign .
docker run --runtime=nvidia --shm-size="16g" -p 8000:8000 -it interiordesign --stableDiffusionModelName sd1.5 --controlnetMethod depth --dtype fp16 --host 0.0.0.0 --port 8000 --checkInput
```

### Auto Installation with Docker
```
bash start.sh [sd1.5] [canny - segmentation - depth - hed - mlsd] [none - fp16 - bf16] [0.0.0.0] [8000]
bash start.sh sd1.5 depth fp16 0.0.0.0 8000
```