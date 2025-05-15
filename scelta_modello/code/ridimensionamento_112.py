from PIL import Image
import os

# === CONFIGURAZIONE ===
input_dir = r"C:\Users\marco\Desktop\Marco\Universita\Magistrale\FVAB\prog\scelta_modello\outputs_finale\Realistic_Vision_V5.0_noVAE"
output_dir = r"C:\Users\marco\Desktop\Marco\Universita\Magistrale\FVAB\prog\magface\inference\img"
temp_list_path = os.path.join(output_dir, "tempimg.list")
img_list_path = os.path.join(output_dir, "img.list")
img_dir_relative = "img"

# === CREA CARTELLA DI DESTINAZIONE ===
os.makedirs(output_dir, exist_ok=True)

# === LEGGI I NOMI DEI FILE DA tempimg.list ===
with open(temp_list_path, "r", encoding="utf-8") as f:
    image_names = [line.strip() for line in f if line.strip()]

# === INIZIALIZZA LISTA DEI PERCORSI RELATIVI ===
new_lines = []

# === LOOP SUI FILE ===
for filename in image_names:
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    if not os.path.exists(input_path):
        print(f"⚠️ File non trovato: {input_path}")
        continue

    # Ridimensionamento
    img = Image.open(input_path)
    img = img.resize((112, 112))
    img.save(output_path)

    # Scrittura della riga in img.list
    new_lines.append(os.path.join(img_dir_relative, filename) + "\n")

# === SCRITTURA FILE img.list ===
with open(img_list_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"\nImmagini ridimensionate e file 'img.list' creato con {len(new_lines)} righe in: {img_list_path}")