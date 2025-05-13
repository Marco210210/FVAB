from PIL import Image
import os

# Percorso della cartella con le immagini originali
input_dir = r"C:\Users\marco\Desktop\Marco\Universita\Magistrale\FVAB\prog\outputs_finale\Realistic_Vision_V5.0_noVAE"  # Aggiungi r per una stringa raw
# Percorso della cartella di destinazione per le immagini ridimensionate
output_dir = r"C:\Users\marco\Desktop\Marco\Universita\Magistrale\FVAB\prog\outputs_finale\Realistic_Vision_V5.0_noVAE_112"  # Usa stringa raw per il percorso

# Crea la cartella di destinazione se non esiste
os.makedirs(output_dir, exist_ok=True)

# Loop attraverso tutte le immagini nella cartella di input
for root, dirs, files in os.walk(input_dir):
    for file in files:
        # Considera solo i file immagine (ad esempio, PNG e JPEG)
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            image_path = os.path.join(root, file)
            img = Image.open(image_path)

            # Ridimensiona l'immagine a 112x112
            img = img.resize((112, 112))  # Modifica questa dimensione se necessario

            # Salva l'immagine ridimensionata nella cartella di destinazione
            img.save(os.path.join(output_dir, file))
            print(f"Immagine ridimensionata salvata in {os.path.join(output_dir, file)}")
