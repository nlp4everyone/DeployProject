from diffusers import StableDiffusionPipeline
import torch
import time

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
beginTime = time.time()
pipe = pipe.to("cuda")

prompt = "a corgi dog is lending on the street"
image = pipe(prompt).images[0]
endTime = time.time() - beginTime
print(endTime)

image.save("astronaut_rides_horse.png")