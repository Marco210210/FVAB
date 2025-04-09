
from diffusers import StableDiffusionPipeline
import torch
import warnings
warnings.filterwarnings("ignore")

# Carica il modello RealisticVision_V2
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Sovrascrivi la funzione di controllo NSFW
def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)  # Restituisce una lista di 'False' per ogni immagine

prompt = ("headshot portrait of a young woman, real life, shot on iPhone, realistic background, HD, HDR color, 4k, natural lighting, photography, Facebook"
"Instagram, Pexels, Flickr, Unsplash, 50mm, 85mm, #wow, AMAZING, epic details, epic, beautiful face, fantastic, cinematic, dramatic lighting")

negative_prompt = "cartoon, blurry, partial face, deformed, artistic filters, black and white, shadows, cropped head, unrealistic, painting, illustration"

# Genera l'immagine con il filtro NSFW disattivato
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    height=1024, width=768,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

# Salva l'immagine
image.save("Realistic_Vision_V5.1_noVAE.png")
