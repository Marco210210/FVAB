from diffusers import AutoPipelineForText2Image
import torch

# Carica il modello SDXL con la pipeline appropriata
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")

prompt = ("headshot portrait of a young woman, real life, shot on iPhone, realistic background, HD, HDR color, 4k, natural lighting, photography, Facebook"
"Instagram, Pexels, Flickr, Unsplash, 50mm, 85mm, #wow, AMAZING, epic details, epic, beautiful face, fantastic, cinematic, dramatic lighting")

# Negativo per evitare artefatti
negative_prompt = "cartoon, blurry, partial face, deformed, artistic filters, black and white, shadows, cropped head, unrealistic, painting, illustration"

# Genera l'immagine con SDXL
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,  # Mantieni lo stesso numero di passi di inferenza
    guidance_scale=7.5  # Puoi regolare questo valore per un miglior risultato
).images[0]

# Salva l'immagine generata
image.save("sdxl_turbo_image4.png")
