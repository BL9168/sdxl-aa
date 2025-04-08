from cog import BasePredictor, Input, Path
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from typing import List, Optional
import os
import numpy as np
from transformers import CLIPImageProcessor

class Predictor(BasePredictor):
    def setup(self):
        print("Loading AAM XL AnimeMix model...")
        self.pipe = DiffusionPipeline.from_pretrained(
            "Lykon/AAM_XL_AnimeMix",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        self.pipe.to("cuda")

        print("Setting up img2img pipeline...")
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            text_encoder_2=self.pipe.text_encoder_2,
            tokenizer=self.pipe.tokenizer,
            tokenizer_2=self.pipe.tokenizer_2,
            unet=self.pipe.unet,
            scheduler=self.pipe.scheduler,
        )
        self.img2img_pipe.to("cuda")

        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def load_image(self, path):
        return load_image(path).convert("RGB")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for image generation"),
        image: Optional[Path] = Input(description="Image for img2img mode", default=None),
        width: int = Input(description="Width of output image", default=1024),
        height: int = Input(description="Height of output image", default=1024),
        prompt_strength: float = Input(description="Prompt strength for img2img", ge=0.0, le=1.0, default=0.7),
        guidance_scale: float = Input(description="CFG scale", ge=1, le=20, default=6.0),
        num_inference_steps: int = Input(description="Steps", ge=10, le=50, default=30),
        seed: Optional[int] = Input(description="Seed for reproducibility", default=None),
    ) -> List[Path]:

        generator = torch.Generator("cuda")
        if seed is not None:
            generator.manual_seed(seed)

        pipe = self.pipe
        extra_args = {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        if image:
            input_image = self.load_image(image)
            pipe = self.img2img_pipe
            extra_args["image"] = input_image
            extra_args["strength"] = prompt_strength
        else:
            extra_args["width"] = width
            extra_args["height"] = height

        output = pipe(**extra_args)

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
