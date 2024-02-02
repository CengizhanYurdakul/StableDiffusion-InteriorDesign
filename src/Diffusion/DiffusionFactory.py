from diffusers import ControlNetModel
from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler

class StableDiffusionFactory:
    def __init__(self, args) -> None:
        self.args = args
    
    def callModel(self, controlNet:ControlNetModel) -> StableDiffusionControlNetPipeline:
        """
        initialises the Stable Diffusion model (downloads it if necessary) according 
        to the method we specify from diffusers.ControlNetModel. When a new method is added, 
        its model initialization is implemented here.

        Args:
            controlNet (ControlNetModel): ControlNet model to be fused into the Stable Diffusion pipeline.

        Raises:
            Exception: If you try to initialize a Stable Diffusion method that has not been implemented 
            before, you will end up here.

        Returns:
            StableDiffusionControlNetPipeline: Stable Diffusion model that will process the image according
            to the incoming input image and prompt.
        """

        if self.args.stableDiffusionModelName == "sd1.5":
            pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlNet, torch_dtype=self.args.dtype).to("cuda")
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        else:
            raise Exception("Sorry, Stable Diffusion %s method not implemented yet!" % self.args.stableDiffusionModelName) 
        print("[StableDiffusion] - Variables initialized!")
        return pipe