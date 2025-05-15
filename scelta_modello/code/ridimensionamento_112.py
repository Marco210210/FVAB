from PIL import Image
import os

# === CONFIGURAZIONE ===
input_dir = r"C:\Users\marco\Desktop\Marco\Universita\Magistrale\FVAB\prog\outputs_finale\Realistic_Vision_V5.0_noVAE"
output_dir = r"C:\Users\marco\Desktop\Marco\Universita\Magistrale\FVAB\prog\magface\inference\img"
img_list_path = os.path.join(output_dir, "img.list")
img_dir_relative = "img"

# === CREA CARTELLA DI DESTINAZIONE ===
os.makedirs(output_dir, exist_ok=True)

# === INIZIALIZZA LISTA DEI FILE ===
new_lines = []

# === LOOP IMMAGINI ===
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_dir, file)

            # Ridimensionamento
            img = Image.open(input_path)
            img = img.resize((112, 112))
            img.save(output_path)

            # Aggiungi il percorso relativo alla lista
            new_lines.append(os.path.join(img_dir_relative, file) + "\n")

# === SCRITTURA FILE img.list ===
with open(img_list_path, 'w') as f:
    f.writelines(new_lines)

print(f"\nImmagini ridimensionate e file 'img.list' creato con {len(new_lines)} righe in: {img_list_path}")
