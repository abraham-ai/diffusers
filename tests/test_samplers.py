import torch
print(torch.__file__)
print(torch.__version__)
from diffusers import StableDiffusionPipeline

import numpy as np
import matplotlib.gridspec as gridspec

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")


from diffusers import LMSDiscreteScheduler, EulerDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler, KDPM2DiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler

schedulers = [
    LMSDiscreteScheduler.from_config(pipe.scheduler.config), 
    EulerDiscreteScheduler.from_config(pipe.scheduler.config),
    DPMSolverMultistepScheduler.from_config(pipe.scheduler.config),
    KDPM2DiscreteScheduler.from_config(pipe.scheduler.config),
    PNDMScheduler.from_config(pipe.scheduler.config),
    ]
prompt = "a photo of an astronaut riding a horse on mars"


n_steps_options = [25]

generators = [torch.Generator(device="cuda").manual_seed(0) for i in range(len(n_steps_options))]

images = []
for scheduler in schedulers:
    pipe.scheduler = scheduler
    for n_steps, generator in zip(n_steps_options, generators):
        image = pipe(prompt, 
                generator = generator,
                height = 512,
                width = 512,
                num_inference_steps = n_steps,
                guidance_scale = 8,
                #negative_prompt: "",
                ).images[0]
        images.append(image)

# Create a grid of all the images, plotting the name of the sampler above each column
# and the number of steps to the left of each row:

import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=len(schedulers), ncols=len(n_steps_options), figsize=(20, 20))

for i, image in enumerate(images):
    ax = axes[i // len(n_steps_options), i % len(n_steps_options)]
    ax.imshow(image)
    #ax.axis("off")

row_headers = [scheduler.__class__.__name__ for scheduler in schedulers]
col_headers = [f"{n_steps} steps" for n_steps in n_steps_options]

for ax, col in zip(axes[0], col_headers):
    ax.set_title(col)

for ax, row in zip(axes[:,0], row_headers):
    ax.set_ylabel(row, rotation=0, size='large')

plt.savefig("test.png")