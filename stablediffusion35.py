import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = ("headshot portrait of a young woman, real life, shot on iPhone, realistic background, HD, HDR color, 4k, natural lighting, photography, Facebook"
"Instagram, Pexels, Flickr, Unsplash, 50mm, 85mm, #wow, AMAZING, epic details, epic, beautiful face, fantastic, cinematic, dramatic lighting")

negative_prompt = "cartoon, blurry, partial face, deformed, artistic filters, black and white, shadows, cropped head, unrealistic, painting, illustration"

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=40,
    guidance_scale=4.5,
).images[0]
image.save("stable-diffusion-3.5-medium2.png")