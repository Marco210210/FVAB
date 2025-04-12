from diffusers import AutoPipelineForText2Image
import torch

# Carica il modello SDXL con la pipeline appropriata
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")

# Prompt più dettagliato per ottenere una qualità simile a quella di SD 3.5 Medium
prompt = ("headshot portrait of a young woman, real life, shot on iPhone, realistic background, HD, HDR color, 4k, natural lighting, photography, Facebook"
"Instagram, Pexels, Flickr, Unsplash, 50mm, 85mm, #wow, AMAZING, epic details, epic, beautiful face, fantastic, cinematic, dramatic lighting")

# Negativo per evitare artefatti
negative_prompt = "cartoon, blurry, partial face, deformed, artistic filters, black and white, shadows, cropped head, unrealistic, painting, illustration"

# Genera l'immagine con SDXL
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,  # Maggiore numero di passi per migliorare la qualità
    guidance_scale=7.5  # Aumenta il valore per un miglior allineamento al prompt
).images[0]

# Salva l'immagine generata
image.save("stable-diffusion-2.1-base.png")