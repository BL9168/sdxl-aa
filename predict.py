import torch
from cog import BasePredictor, Input, Path
from typing import List, Optional
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from PIL import Image

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
        ).to("cuda")

        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
        ).to("cuda")

    def load_image(self, image_path: Path) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        return image

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt to guide the image generation"),
        negative_prompt: Optional[str] = Input(description="Negative prompt to avoid certain features", default=None),
        image: Optional[Path] = Input(description="Optional image for img2img mode", default=None),
        width: int = Input(description="Width of output image", default=1024),
        height: int = Input(description="Height of output image", default=1024),
        prompt_strength: float = Input(description="Strength of prompt for img2img", default=0.7),
        guidance_scale: float = Input(description="How strongly the prompt is followed", default=6.0),
        num_inference_steps: int = Input(description="Number of denoising steps", default=20),
        scheduler_name: str = Input(description="Scheduler to use", default="K_EULER_ANCESTRAL", choices=list(SCHEDULERS.keys())),
        seed: Optional[int] = Input(description="Seed for reproducibility", default=None),
    ) -> List[Path]:

        generator = torch.Generator("cuda")
        if seed is not None:
            generator.manual_seed(seed)

        pipe = self.img2img_pipe if image else self.pipe

        # 切換 Scheduler
        scheduler_cls = SCHEDULERS[scheduler_name]
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)

        extra_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        if image:
            input_image = self.load_image(image)
            extra_args["image"] = input_image
            extra_args["strength"] = prompt_strength
        else:
            extra_args["width"] = width
            extra_args["height"] = height

        output = pipe(**extra_args).images

        output_paths = []
        for i, img in enumerate(output):
            out_path = f"/tmp/out-{i}.png"
            img.save(out_path)
            output_paths.append(Path(out_path))

        return output_paths
