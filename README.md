# Régression
Cet exemple vous montre comment :
- Définir un jeu de données personnalisé pour les problèmes de régression. Nous implémentons le
  [California Housing Dataset](https://huggingface.co/datasets/gvlassis/california_housing) depuis
  HuggingFace hub. Le jeu de données est également disponible parmi les jeux de données de régression
  jouets dans sklearn [datasets](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).
- Créer un pipeline de données depuis un jeu de données brut jusqu'à un DataLoader rapide par lots
  avec mise à l'échelle min-max des features.
- Définir un modèle de réseau de neurones simple pour la régression en utilisant les Modules Burn.

> **Note**  
> Cet exemple utilise la bibliothèque [`datasets`](https://huggingface.co/docs/datasets/index)
> de HuggingFace pour télécharger les jeux de données. Assurez-vous d'avoir
> [Python](https://www.python.org/downloads/) installé sur votre ordinateur.

L'exemple peut être exécuté ainsi :
```bash
git clone https://github.com/cia-ulaval/tutoriel-burn.git
cd tutoriel-burn
```

## Étape 1 : entraîner le modèle

Regardez dans `model.rs`, la structure `RegressionModel`. Elle est composée de :
- 1 couche de neurones
- 1 couche d'activation

Les poids et biais de ces neurones sont entraînés avec l'entraînement décrit dans `training.rs`.

> Utilisez le flag `--release` pour vraiment accélérer l'entraînement !

Exécuter avec `ndarray`, sur le CPU :
```bash
cargo run --features ndarray             # Backend CPU NdArray - f32 - fil d'exécution unique
```

Exécuter sur n'importe quelle plateforme (GPU, CPU...) avec le backend `wgpu` :
```bash
cargo run --features wgpu
```

Exécuter sur un GPU NVIDIA avec le backend `cuda` :

**Linux / macOS (bash) :**
```bash
export TORCH_CUDA_VERSION=cu128
cargo run --features cuda
```

**Windows (PowerShell) :**
```powershell
$env:TORCH_CUDA_VERSION = "cu128"
cargo run --features cuda
```

Executer sur un GPU AMD avec le backend `rocm` :
```bash
echo "Utilisation du backend rocm"
cargo run --features rocm                # Backend ROCM
```

Utiliser tch backend (CUDA, Metal):

**Linux / macOS (bash) :**
```bash
export TORCH_CUDA_VERSION=cu128
cargo run --features tch-gpu
```

**Windows (PowerShell) :**
```powershell
$env:TORCH_CUDA_VERSION = "cu128"
cargo run --features tch-gpu
```

## Étape 2 : Jouer avec les hyperparamètres

Essayez de changer des propriétés du modèle dans `model.rs`. Que se passe-t-il si vous changez les choses suivantes ?
- Nombre de neurones dans la couche cachée
- Biais vs aucun biais
- La fonction d'activation (ReLU ? GeLU ? sigmoid ?)

## Étape 3 : ajouter une couche

Essayez d'ajouter une couche de neurones au réseau.
Assurez-vous que tout compile encore !

Si vous êtes rendu ici, bravo ! Utiliser un nouveau language et une librarie complexe est une tâche compliquée. 

Si vous avez le goût d'en faire plus, passez à l'étape 4...

## Étape 4 : aller plus loin

Choisissez un défis parmi les suivants :

**A — Dropout**  
Ajoutez une couche de `Dropout` entre deux couches de neurones. Consultez le [Burn book](https://burn.dev/books/burn/basic-workflow/model.html) pour un exemple. Observez : est-ce que la loss de validation s'améliore par rapport à la loss d'entraînement ?

**B — Changer l'optimiseur**  
Dans `training.rs`, remplacez l'optimiseur actuel par `Adam`. Comparez la vitesse de convergence avec SGD. Lequel atteint une loss plus basse en 10 époques ?
Regardez [ici](https://burn.dev/books/burn/basic-workflow/backend.html?highlight=Adam) pour un exemple.

**C — Sauvegarder et recharger le modèle**  
Après l'entraînement, sauvegardez les poids avec le `NamedMpkFileRecorder` de Burn, puis rechargez-les dans un nouveau binaire pour faire des prédictions sans réentraîner. Regardez [ici](https://burn.dev/books/burn/saving-and-loading.html) pour la documentation.
