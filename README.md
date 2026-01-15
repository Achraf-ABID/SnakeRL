Voici une proposition de fichier **README.md** professionnel, structuré et prêt à l'emploi.

Il est conçu pour mettre en valeur ton travail, expliquer l'algorithme (Dueling DQN) et intégrer parfaitement les images que tu as fournies.

### ⚠️ Étape importante avant de copier le code :
D'après ta capture d'écran, tu as un dossier nommé `Image`. Pour que le README fonctionne, **renomme tes images** et place-les dans ce dossier comme suit :

1.  La capture du jeu (le serpent vert) ➔ **`Image/gameplay.png`**
2.  Les graphiques (courbes bleues et vertes) ➔ **`Image/charts.png`**
3.  Le résumé du terminal (texte blanc sur noir) ➔ **`Image/summary.png`**

---

### Copie ce code dans ton fichier `README.md` :

```markdown
# SnakeRL 🐍 🤖

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Pygame](https://img.shields.io/badge/Pygame-Game%20Engine-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**SnakeRL** est un agent d'Intelligence Artificielle capable d'apprendre à jouer au jeu Snake de zéro en utilisant l'apprentissage par renforcement (Reinforcement Learning).

Le projet implémente un réseau de neurones de type **Dueling Deep Q-Network (Dueling DQN)**, permettant à l'agent d'optimiser ses décisions pour maximiser son score et sa survie.

## 📷 Aperçu du projet

<p align="center">
  <img src="Image/gameplay.png" alt="Snake AI Gameplay" width="600">
</p>

## 🚀 Fonctionnalités

- **Deep Q-Learning (DQN) :** Utilisation de l'expérience replay et d'un réseau cible (target network).
- **Architecture Dueling DQN :** Séparation de l'estimation de la valeur de l'état et de l'avantage de l'action pour une meilleure stabilité.
- **Visualisation en Temps Réel :** Graphiques Matplotlib dynamiques pour suivre :
  - Le score par épisode.
  - La moyenne mobile.
  - La perte (loss) d'entraînement.
- **Persistance :** Sauvegarde automatique du meilleur modèle (`model_record.pth`) et des checkpoints.
- **Modes :**
  - Mode **Entraînement** (Accéléré pour l'apprentissage).
  - Mode **Démo** (Vitesse normale pour observer l'IA jouer).

## 📊 Résultats d'Entraînement

Après environ 500 épisodes, l'IA développe une stratégie solide, évitant les murs et son propre corps tout en chassant la nourriture efficacement.

| Métrique | Valeur atteinte |
| :--- | :--- |
| **Meilleur Score** | **38** 🏆 |
| **Score Moyen** | **8.71** |
| **Épisodes** | **561** |

### Graphiques de performance
Voici l'évolution de l'apprentissage (Score, Moyenne, Perte et Distribution) :

![Training Charts](Image/charts.png)

### Résumé final
![Training Summary](Image/summary.png)

## 🛠️ Installation

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/Achraf-ABID/SnakeRL.git
   cd SnakeRL
   ```

2. **Installer les dépendances :**
   Il est recommandé d'utiliser un environnement virtuel (venv ou conda).
   ```bash
   pip install pygame torch torchvision matplotlib numpy
   ```

## 🎮 Utilisation

Lancez le script principal :

```bash
python SnakeRL.py
```

Le programme vous demandera de choisir un mode :
1. **Entraîner un nouveau modèle :** L'IA commence sans connaissances et apprend au fil du temps.
2. **Démo avec modèle existant :** Charge un fichier `.pth` pour voir l'IA jouer à pleine puissance.

## 🧠 Architecture Technique

### Entrées (State)
Le réseau de neurones prend en entrée un vecteur de 19 valeurs (booléens et flottants) représentant :
- Dangers immédiats (tout droit, droite, gauche).
- Direction actuelle du mouvement.
- Position relative de la nourriture.

### Modèle (Neural Network)
- **Input Layer :** 19 neurones.
- **Hidden Layer :** 256 neurones (ReLU).
- **Dueling Layers :**
  - *Value Stream* : Estime la valeur de l'état actuel.
  - *Advantage Stream* : Estime l'avantage de chaque action.
- **Output :** 3 actions possibles `[Tout droit, Droite, Gauche]`.

## 📂 Structure du projet

```text
SnakeRL/
├── Image/                  # Captures d'écran et graphiques
├── models/                 # Dossier où sont sauvegardés les modèles (.pth)
├── SnakeRL.py              # Code source principal (Jeu + Agent + Entraînement)
└── README.md               # Documentation
```

## 📝 Auteur

**Achraf ABID** - *Développeur & Data Scientist*
```

---

### Pourquoi ce README est professionnel ?

1.  **Badges** : Ils donnent immédiatement les infos techniques (Python, PyTorch).
2.  **Visuels** : Les images sont intégrées avec des balises HTML `<img src="..." width="...">` ou Markdown standard pour s'assurer qu'elles ne soient pas trop grandes à l'écran.
3.  **Tableau des résultats** : Met en avant ton record de 38 points de manière claire.
4.  **Explication technique** : Une section explique *comment* ça marche (inputs, layers), ce qui prouve que tu maîtrises ton code.
5.  **Installation/Usage** : Indispensable pour que d'autres puissent tester ton code.
