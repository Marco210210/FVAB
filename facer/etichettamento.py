import sys
import torch
import facer
import csv
import os

sys.path.append("..")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Cartella delle immagini
img_dir = r"C:\Users\marco\Desktop\Marco\Universita\Magistrale\FVAB\prog\magface\inference\img"

# Modello di rilevamento facciale
face_detector = facer.face_detector("retinaface/mobilenet", device=device)

# Modello per gli attributi facciali
face_attr = facer.face_attr("farl/celeba/224", device=device)
labels = ['image_id'] + face_attr.labels

# Output CSV
output_csv = os.path.join(img_dir, "labels.csv")
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)  # intestazione

    # Scorri tutte le immagini nella cartella
    for img_name in os.listdir(img_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(img_dir, img_name)
            image = facer.hwc2bchw(facer.read_hwc(img_path)).to(device=device)

            # Rilevamento e predizione
            with torch.inference_mode():
                faces = face_detector(image)
                if len(faces["scores"]) == 0:
                    print(f"Nessun volto rilevato in: {img_name}")
                    continue
                faces = face_attr(image, faces)

            # Estrai attributi del primo volto
            face1_attrs = faces["attrs"][0]
            row = [img_name] + [1 if p > 0.5 else -1 for p in face1_attrs]
            writer.writerow(row)

print(f"Etichettamento completato. Risultati salvati in {output_csv}")
