import os
import csv

# Percorso del file .list
input_list_path = r"C:\Users\marco\Desktop\Marco\Universita\Magistrale\FVAB\prog\magface\inference\img\img.list"

# Percorso di output per il CSV finale
output_csv_path = r"C:\Users\marco\Desktop\Marco\Universita\Magistrale\FVAB\prog\magface\inference\img\partition.csv"

# Leggi tutti i nomi immagine e rimuovi il prefisso "img\"
with open(input_list_path, 'r', encoding='utf-8') as f:
    image_filenames = [line.strip().replace("img\\", "").replace("img/", "") for line in f if line.strip()]

# Calcola numero totale e soglie per partizioni
total = len(image_filenames)
train_split = int(total * 0.8)
val_split = int(total * 0.1)

# Scrittura del file CSV
with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_id', 'partition'])  # intestazione

    for i, image_name in enumerate(image_filenames):
        if i < train_split:
            partition = 0  # train
        elif i < train_split + val_split:
            partition = 1  # validation
        else:
            partition = 2  # test
        writer.writerow([image_name, partition])

print(f"✅ Creato 'partition.csv' con {total} righe in:\n{output_csv_path}")
