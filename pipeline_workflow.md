# 🧪 Workflow per la Pipeline di Etichettamento e Riaddestramento

Questa guida descrive passo-passo l'intero processo manuale da seguire per la generazione, etichettamento e riaddestramento del modello Slim-CNN.

---

## 1. ✏️ Scrittura dei Prompt
Scrivere i prompt nel file:
```
prog/scelta_modello/prompt_test2.txt
```

---

## 2. 🖼️ Generazione delle Immagini
Lanciare il file Python per generare le immagini con il modello scelto:
```
prog/scelta_modello/code/generazione_modello_scelto.py
```

---

## 3. 📏 Ridimensionamento Immagini + Creazione `img.list`
Eseguire il ridimensionamento a 112x112 e generare il file `img.list`:
```
prog/scelta_modello/code/ridimensionamento_112.py
```

---

## 4. 🧬 Generazione delle Feature
Portarsi nella cartella:
```
prog/magface/inference
```
e da lì eseguire il comando:
```bash
python gen_feat.py --inf_list img/img.list --feat_list img/feat.list --resume ../models/magface_epoch_00025.pth
```

---

## 5. 📊 Valutazione delle Immagini
Aprire ed eseguire il notebook:
```
prog/magface/inference/examples.ipynb
```

---

## 6. 🏷️ Etichettamento delle Immagini
Eseguire lo script:
```
prog/facer/etichettamento.py
```

---

## 7. 🧩 Partizionamento del Dataset
Creare il file CSV con le partizioni (train/val/test) eseguendo:
```
prog/pytorch-slim-cnn/partizionamento.py
```

---

## 8. 🧠 Riaddestramento del Modello Slim-CNN
Lanciarsi nella cartella:
```
prog/pytorch-slim-cnn
```

E avviare il riaddestramento con:
```bash
python train_new.py --batch_size 16 --num_workers 0
```

---

✅ Fine del processo manuale. Ricontrollare i file `labels.csv` e `partition.csv` prima dell'addestramento finale.
