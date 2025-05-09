import torch
from slimnet import SlimNet
from torchvision import transforms
from PIL import Image
import numpy as np

model_path = 'checkpoints/model_20.pth'
model = SlimNet(filter_count_values=[16, 32, 48, 64], initial_conv=[96, 7, 2], num_classes=40, depth_multiplier=1)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval() 

transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  
])

img_path = r'Realistic_Vision_V5.0_noVAE-512\1_Realistic_Vision_V5.0_noVAE_1.png'  # Sostituisci con il percorso dell'immagine
image = Image.open(img_path).convert('RGB')

image = transform(image).unsqueeze(0)  
image = image.to(device) 

with torch.no_grad():
    logits = model(image) 

# funzione sigmoide per ottenere probabilità
sigmoid_logits = torch.sigmoid(logits)

predictions = sigmoid_logits > 0.5  # soglia di 0.5 per classificare
predictions = predictions.cpu().numpy()

# classi previste
predicted_labels = np.where(predictions[0])[0] 
print("Predicted classes:", predicted_labels)

# probabilità per ciascuna classe
probabilities = sigmoid_logits.cpu().numpy()[0]
print("Class probabilities:", probabilities)
