from diffusers import ControlNetModel

class ControlNetFactory:
    def __init__(self, args) -> None:
        self.args = args
        
        self.callModel()
    
    def callModel(self) -> ControlNetModel:
        """
        initialises the ControlNet model (downloads it if necessary) according 
        to the method we specify from diffusers.ControlNetModel. When a new method is added, 
        its model initialization is implemented here.

        Raises:
            Exception: If you try to initialize a ControlNet method that has not been implemented 
            before, you will end up here.

        Returns:
            ControlNetModel: ControlNet model built in accordance with the specified method
        """
        
        if self.args.controlnetMethod == "canny":
            self.model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=self.args.dtype)
        elif self.args.controlnetMethod == "segmentation":
            self.model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=self.args.dtype)
        elif self.args.controlnetMethod == "depth":
            self.model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=self.args.dtype)
        elif self.args.controlnetMethod == "hed":
            self.model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=self.args.dtype)
        elif self.args.controlnetMethod == "mlsd":
            self.model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd", torch_dtype=self.args.dtype)
        else:
            raise Exception("Sorry, controlnet %s method not implemented yet!" % self.args.controlnetMethod)
        
        print("[ControlNet] - Variables initialized!")
        return self.model
