import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Charger le tokenizer et le modèle alternatif
SONAR_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Modèle alternatif
tokenizer = AutoTokenizer.from_pretrained(SONAR_MODEL_NAME)
sonar_model = AutoModel.from_pretrained(SONAR_MODEL_NAME)

# Définir le modèle TwoTowerLCM
class TwoTowerLCM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers):
        super(TwoTowerLCM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Couche de pré-traitement pour les embeddings d'entrée
        self.pre_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)  # Ajouter du dropout
        )
        
        # Contextualizer : TransformerEncoder pour encoder le contexte
        self.contextualizer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,  # Ajouter du dropout
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Denoiser : TransformerDecoder pour prédire l'embedding suivant
        self.denoiser = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,  # Ajouter du dropout
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Couche de post-traitement pour générer l'embedding final
        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh(),  # Normalisation des embeddings
            nn.Dropout(0.1)  # Ajouter du dropout
        )
    
    def forward(self, context_embeddings):
        # Transformation des embeddings d'entrée
        x = self.pre_net(context_embeddings)  # (batch_size, seq_len, hidden_dim)
        
        # Encodage du contexte avec le contextualizer
        context_encoded = self.contextualizer(x)  # (batch_size, seq_len, hidden_dim)
        
        # Génération de l'embedding suivant avec le denoiser
        noise = torch.randn(context_embeddings.size(0), 1, self.hidden_dim)  # (batch_size, 1, hidden_dim)
        next_embedding = self.denoiser(noise, context_encoded)  # (batch_size, 1, hidden_dim)
        
        # Transformation finale
        next_embedding = self.post_net(next_embedding.squeeze(1))  # (batch_size, embedding_dim)
        
        return next_embedding

# Fonction pour générer des embeddings
def get_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Utiliser le dernier état caché (CLS token) comme embedding de la phrase
    embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, embedding_dim)
    return embeddings

# Paramètres du modèle
embedding_dim = 384  # Dimension des embeddings du modèle alternatif
hidden_dim = 1024    # Augmenter la dimension cachée
num_layers = 8       # Augmenter le nombre de couches

# Instanciation du modèle
model = TwoTowerLCM(embedding_dim, hidden_dim, num_layers)

# Exemple de conversations cohérentes
conversations = [
    ["Bonjour, comment ça va ?", "Ça va bien, merci !", "Qu'as-tu fait aujourd'hui ?", "J'ai travaillé sur un projet."],
    ["Salut, tu viens ce soir ?", "Oui, je serai là.", "On se voit à 20h ?", "Parfait, à tout à l'heure !"],
    ["C'est quoi ton film préféré ?", "J'adore Inception.", "Pourquoi ?", "Parce que l'histoire est fascinante."],
    ["Tu as vu la météo aujourd'hui ?", "Oui, il va pleuvoir toute la journée.", "Dommage, j'espérais faire une balade.", "On peut toujours aller au cinéma !"],
    ["Quel est ton plat préféré ?", "J'adore les sushis.", "Moi aussi !", "On devrait en manger ce week-end."]
]

# Générer les embeddings pour chaque conversation
context_embeddings = []
target_embeddings = []
for conv in conversations:
    # Convertir les phrases en embeddings
    embeddings = get_embeddings(conv, tokenizer, sonar_model)  # (4, 384)
    
    # Les 3 premières phrases sont le contexte
    context_embeddings.append(embeddings[:-1])  # (3, 384)
    
    # La dernière phrase est la cible
    target_embeddings.append(embeddings[-1])    # (384,)

# Convertir en tenseurs PyTorch
context_embeddings = torch.stack(context_embeddings)  # (5, 3, 384)
target_embeddings = torch.stack(target_embeddings)    # (5, 384)

# Définir la fonction de perte et l'optimiseur
criterion = nn.MSELoss()  # Perte de régression
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Réduire le taux d'apprentissage

# Boucle d'entraînement
model.train()  # Passer en mode entraînement
for epoch in range(20):  # Augmenter le nombre d'époques
    optimizer.zero_grad()  # Réinitialiser les gradients
    
    # Prédiction
    next_embedding = model(context_embeddings)
    
    # Calcul de la perte
    loss = criterion(next_embedding, target_embeddings)
    
    # Rétropropagation et mise à jour des poids
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Sauvegarder le modèle entraîné
torch.save(model.state_dict(), "two_tower_lcm.pth")
print("Modèle sauvegardé avec succès.")

# Charger le modèle sauvegardé
model = TwoTowerLCM(embedding_dim, hidden_dim, num_layers)
model.load_state_dict(torch.load("two_tower_lcm.pth"))
model.eval()  # Passer en mode évaluation
print("Modèle chargé avec succès.")

# Fonction pour évaluer le modèle
def evaluate_model(model, tokenizer, sonar_model, context_phrases, target_phrase):
    # Convertir les phrases en embeddings
    context_embeddings = get_embeddings(context_phrases, tokenizer, sonar_model)  # (seq_len, 384)
    context_embeddings = context_embeddings.unsqueeze(0)  # (1, seq_len, 384)
    
    # Convertir la phrase cible en embedding
    target_embedding = get_embeddings([target_phrase], tokenizer, sonar_model)  # (1, 384)
    
    # Prédiction de l'embedding suivant
    with torch.no_grad():
        next_embedding = model(context_embeddings)  # (1, 384)
    
    # Calculer la similarité cosinus entre l'embedding prédit et l'embedding cible
    similarity = cosine_similarity(next_embedding.cpu().numpy(), target_embedding.cpu().numpy())
    return similarity[0][0]

# Tester le modèle avec une nouvelle séquence
new_context = ["Bonjour, comment ça va ?", "Ça va bien, merci !", "Qu'as-tu fait aujourd'hui ?"]
target_phrase = "J'ai travaillé sur un projet."
similarity = evaluate_model(model, tokenizer, sonar_model, new_context, target_phrase)
print(f"Similarité cosinus entre l'embedding prédit et l'embedding cible : {similarity}")