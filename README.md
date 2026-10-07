# Deep Positive-Unlabeled Anomaly Detection for Contaminated Unlabeled Data
This is a pytorch implementation of the following paper [[arXiv]](https://arxiv.org/abs/2405.18929):
```
@misc{takahashi2024deep,
      title={Deep Positive-Unlabeled Anomaly Detection for Contaminated Unlabeled Data}, 
      author={Hiroshi Takahashi and Tomoharu Iwata and Atsutoshi Kumagai and Yuuki Yamanaka},
      year={2024},
      eprint={2405.18929},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2405.18929}, 
}
```
Please read [LICENCE.md](LICENCE.md) before reading or using the files.


## Prerequisites
- Please install `python>=3.10`, `numpy`, `scipy`, `torch`, `torchvision`, `scikit_learn`, and `matplotlib`
- Please also see `requirements.txt`


## Datasets
All datasets will be downloaded when first used.


## Usage

### for MNIST, FashionMNIST, SVHN and CIFAR10
```
usage: main.py [-h] [--dataset DATASET] [--normal_class NORMAL_CLASS]
               [--unseen_anomaly UNSEEN_ANOMALY] [--algorithm ALGORITHM]
               [--alpha ALPHA] [--n_epoch N_EPOCH]
               [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
               [--seed SEED]
```
- You can choose the `dataset` from following datasets: 
  - `MNIST`: handwritten digits
  - `FashionMNIST`: fashion product images
  - `SVHN`: house number digits
  - `CIFAR10`: animal and vehicle images
- You can choose the `normal_class` from 0 to 9
- You can choose the `unseen_anomaly` from 0 to 9
- You can choose the `algorithm` from following algorithms:
  - `IF`: Isolation Forest
  - `AE`: Autoencoder
  - `DeepSVDD`: DeepSVDD
  - `LOE`: Latent Outlier Exposure
  - `ABC`: Autoencoding Binary Classifier
  - `DeepSAD`: Deep Semi-Supervised Anomaly Detection
  - `SOEL`: Semi-supervised Outlier Exposure with a Limited labeling budget
  - `PU`: PU Learning Classifier
  - `PUAE`: Our approach with AE
  - `PUSVDD`: Our approach with DeepSVDD
- You can change the `alpha`, the hyperparameter of PU learning-based approaches
- You can change the random `seed` of the training and `n_epoch`, `learning_rate`, and `batch_size` of the optimizer


### for toy dataset
```
usage: toy.py [-h] [--algorithm ALGORITHM] [--alpha ALPHA] [--n_epoch N_EPOCH]
              [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
              [--seed SEED]
```
- You can choose the `algorithm` from following algorithms:
  - `AE`: Autoencoder
  - `DeepSVDD`: DeepSVDD
  - `LOE`: Latent Outlier Exposure
  - `ABC`: Autoencoding Binary Classifier
  - `DeepSAD`: Deep Semi-Supervised Anomaly Detection
  - `SOEL`: Semi-supervised Outlier Exposure with a Limited labeling budget
  - `PU`: PU Learning Classifier
  - `PUAE`: Our approach with AE
  - `PUSVDD`: Our approach with DeepSVDD
- You can change the `alpha`, the hyperparameter of PU learning-based approaches
- You can change the random `seed` of the training and `n_epoch`, `learning_rate`, and `batch_size` of the optimizer


## Example
MNIST experiment (normal: 1 / unseen: 0) with our approach:
```
python main.py --dataset MNIST --normal_class 1 --unseen_anomaly 0 --algorithm PUSVDD
```


## TAVIL latent viewer

The repository now also contains the recording-grouped TAVIL implementation in
`tavil/` and an interactive baseline checkpoint viewer in
`tavil/latent_view.py`.  The viewer keeps one ensemble member's latent
geometry separate from the six-member ensemble score, supports AE=0
initialization and AE>0 pre-PU states, PCA3 projection for q>3, recording
selection, ranking inversions and an orthographic Plotly scene.

Install the optional viewer dependencies with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_visualization.txt
```

The controlled linear/nonlinear × AE/AE+PU experiments live separately in the
private companion repository `joeldarnes/PUSHVDD2`; they are intentionally
not mixed into this baseline repository.
