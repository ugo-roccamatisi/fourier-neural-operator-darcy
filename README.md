# TP : Fourier Neural Operator sur Darcy Flow

Travaux pratiques d'apprentissage d'opérateurs : on apprend l'opérateur qui associe un champ de perméabilité $a(x,y)$ au champ de pression $u(x,y)$ solution de l'équation de Darcy, $-\nabla \cdot (a \nabla u) = f$. Le notebook s'inspire du notebook *Training on Darcy Flow* du bootcamp [NeuralOperator](https://github.com/neuraloperator/neuraloperator).

## Contenu

- Rappels sur l'écoulement de Darcy et sur les Fourier Neural Operators (FNO, TFNO)
- Chargement du dataset Darcy, visualisation des champs et du flux $\mathbf{v} = -a \nabla u$
- Losses et métriques : L2 relative et H1
- Baseline U-Net (CNN champ vers champ) vs (T)FNO
- Test de généralisation en résolution (zero-shot super-resolution : entraînement en 16×16, test en 32×32)
- Ablations sur 2 graines : loss d'entraînement, `n_modes`, capacité, TFNO vs FNO, budget de données
- Test physique sur le flux et bonus Navier-Stokes (rollout temporel)

## Principaux résultats

| Modèle | params | test16 L2 | test32 L2 (zero-shot) |
|---|---:|---:|---:|
| U-Net | 3.12 M | 0.266 | 0.776 |
| TFNO | 23 k | 0.187 | 0.201 |
| TFNO, 1000 éch., 50 époques | 23 k | 0.077 | 0.106 |

Le TFNO atteint une erreur plus faible avec environ 135 fois moins de paramètres et se dégrade peu en changeant de résolution, alors que le U-Net double son erreur en zero-shot. Tableau complet et interprétation dans le notebook.

## Structure

```
ODL_lab_FNO_2026.ipynb   notebook du TP (exécuté)
requirements.txt
```

## Lancer le notebook

```bash
pip install -r requirements.txt
jupyter notebook ODL_lab_FNO_2026.ipynb
```

Le dataset Darcy est téléchargé automatiquement par `neuraloperator`. Le notebook tourne sur CPU ; sur GPU, il suffit d'augmenter `n_train` et `n_epochs`.
