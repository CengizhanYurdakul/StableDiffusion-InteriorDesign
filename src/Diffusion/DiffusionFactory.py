from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler

class StableDiffusionFactory:
    def __init__(self, args) -> None:
        self.args = args
    
    def callModel(self, controlNet):
        if self.args.stableDiffusionModelName == "sd1.5":
            pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlNet, torch_dtype=self.args.dtype).to("cuda")
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        else:
            raise Exception("Sorry, Stable Diffusion %s method not implemented yet!" % self.args.stableDiffusionModelName) 
        print("[StableDiffusion] - Variables initialized!")
        return pipe