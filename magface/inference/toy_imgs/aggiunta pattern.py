import os  # Aggiungi questa riga all'inizio del tuo script

# Percorso della cartella toy_imgs
toy_imgs_dir = "toy_imgs"

# Carica il file img.list originale
with open(r'C:\Users\marco\Desktop\Marco\Universita\Magistrale\FVAB\prog\magface\inference\toy_imgs\img.list', 'r') as f:
    lines = f.readlines()

# Modifica i percorsi nel file
new_lines = []
for line in lines:
    # Aggiungi il prefisso toy_imgs/ ai percorsi relativi
    img_name = line.strip()
    new_img_path = os.path.join(toy_imgs_dir, img_name)
    new_lines.append(new_img_path + "\n")

# Salva il nuovo file img.list con i percorsi modificati
with open(r'C:\Users\marco\Desktop\Marco\Universita\Magistrale\FVAB\prog\magface\inference\toy_imgs', 'w') as f:
    f.writelines(new_lines)

print("Nuovo file img.list creato come 'new_img.list'")
