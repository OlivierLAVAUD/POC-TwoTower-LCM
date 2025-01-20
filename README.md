# 🚀 Concept du Two-Tower LCM (Large Concept Model) 🧠

Le **Two-Tower LCM** est un modèle de génération de texte qui opère dans un **espace sémantique de haut niveau**, plutôt que de travailler directement au niveau des mots ou des tokens (comme le font les modèles de langage traditionnels). L'idée principale est de modéliser des **concepts** abstraits, qui peuvent représenter des phrases ou des idées complètes, indépendamment de la langue ou de la modalité (texte, parole, etc.).

---

## 🌟 Pourquoi ce modèle est différent ?

### 1. **Niveau d'abstraction** 🎯
- Les modèles de langage traditionnels (comme GPT) fonctionnent au niveau des **tokens** (mots ou sous-mots), ce qui limite leur capacité à raisonner à un niveau conceptuel plus élevé.
- Le **Two-Tower LCM** opère au niveau des **phrases** ou des **concepts**, ce qui permet une meilleure cohérence dans la génération de textes longs et une meilleure généralisation à différentes langues.

### 2. **Deux tours (Two-Tower)** 🏰
- Le modèle est composé de deux parties principales :
  1. **Contextualizer** : Il encode le contexte (les phrases précédentes) en un espace sémantique.
  2. **Denoiser** : Il prédit la phrase suivante (ou le concept suivant) en se basant sur le contexte encodé.
- Cette séparation permet au modèle de mieux capturer les dépendances contextuelles et de générer des phrases cohérentes.

### 3. **Espace sémantique** 🌐
- Le modèle utilise des **embeddings de phrases** (représentations vectorielles de phrases) pour encoder et prédire les concepts.
- Ces embeddings sont générés par un modèle pré-entraîné (comme `sentence-transformers/all-MiniLM-L6-v2`), ce qui permet de travailler dans un espace sémantique riche et multilingue.

### 4. **Généralisation zéro-shot** 🎯
- Grâce à l'utilisation d'un espace sémantique indépendant de la langue, le modèle peut être appliqué à des langues ou des modalités qu'il n'a jamais vues pendant l'entraînement, sans nécessiter de données supplémentaires.

---


## 🛠️ Installation (Windows)

### Préparer l'environnement Python

1. **Clonez le dépôt et installez les dépendances :**

   ```bash
   git clone https://github.com/OlivierLAVAUD/POC-TwoTower-LCM.git
   cd POC-TwoTower-LCM
   pip install uv
   uv venv
   .venv\Scripts\activate
   uv pip install -e .

2. **Executez l'application:**
    ```bash
    uv run app.py



