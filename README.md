# Régression
Cet exemple vous montre comment :
- Définir un jeu de données personnalisé pour les problèmes de régression. Nous implémentons le
  [California Housing Dataset](__https://huggingface.co/datasets/gvlassis/california_housing__) depuis
  HuggingFace hub. Le jeu de données est également disponible parmi les jeux de données de régression
  jouets dans sklearn[datasets](__https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset__).
- Créer un pipeline de données depuis un jeu de données brut jusqu'à un DataLoader rapide par lots
  avec mise à l'échelle min-max des features.
- Définir un modèle de réseau de neurones simple pour la régression en utilisant les Modules Burn.

> **Note**  
> Cet exemple utilise la bibliothèque [`datasets`](__https://huggingface.co/docs/datasets/index__)
> de HuggingFace pour télécharger les jeux de données. Assurez-vous d'avoir
> [Python](__https://www.python.org/downloads/__) installé sur votre ordinateur.

L'exemple peut être exécuté ainsi :

```bash
git clone https://github.com/cia-ulaval/tutoriel-burn.git
cd tutoriel-burn
```

> Utilisez le flag --release pour vraiment accélérer l'entraînement !

Executer avec `ndarray`, sur le CPU :
```bash
echo "Utilisation du backend ndarray"
cargo run --features ndarray             # Backend CPU NdArray - f32 - fil d'exécution unique
```

Executer sur un GPU NVIDIA avec le backend `cuda` :
```bash
echo "Utilisation du backend cuda"
export TORCH_CUDA_VERSION=cu128          # Définir la version de cuda
cargo run --features cuda                # Backend CUDA
```

Executer sur un GPU AMD avec le backend `rocm` :
```bash
echo "Utilisation du backend rocm"
cargo run --features rocm                # Backend ROCM
```

Executer sur n'importe quelle autre plateforme avec le backend `wgpu` :
```bash
echo "Utilisation du backend wgpu"
cargo run --features wgpu
```
