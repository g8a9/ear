# Entropy-based Attention Regularization Frees Unintended Bias Mitigation from Lists

## Project structure

The data used in this study is in `data`.
The code is organized in python scripts (training and evaluation of models), bash scripts to run experiments, and jupyter notebooks.

The main files are the following:
- `train_bert.py`: use this script to train any bert-based model starting from HuggingFace checkpoints.
- `evaluate_model.py`: use this script to evaluate a model either on a test set or a synthetic evaluation set.
- `./bash/train_model_10_seeds.sh`: run this script to train a model with 10 different initialization runs (see below for usage).
- `./bash/train_model_EAR_10_seeds.sh`: run this script to train a model with 10 different initialization runs and apply Entropy Attention Regularization (EAR) to the model (see below for usage).

Please find all the accepted parameters running `python train_bert.py --help` or `python evaluate_model.py --help`. 

## Getting started

The following are the basic steps to setup our environment and replicate our results.

## Getting the data sets

Please follow these instructions to retrive the presented dataset:

- Misogyny (EN): the dataset is not publicly available. Please fill [this form](https://docs.google.com/forms/d/e/1FAIpQLSevs4Ji3dNmK5CxyulYG-PxX3U10-RgDrPpMKPRjtI81f0yaQ/viewform) to submit a request to the authors.
- Misogyny (IT): the dataset is not publicly available. Please fill [this form](https://forms.gle/uFF3sAtMMqayiDiz9) to submit a request to the authors.
- Multilingual and Multi-Aspect (MlMA): the dataset is available online. In `data`, we provide our splitfiles with the additional binary "hate" column used in our experiments.

For the sake of simplicty, we have assigned short names to each data set. Please find them and how to use them in [dataset.py](./dataset.py).

## Dependencies

You'll need a working Python environment to run the code. 
The required dependencies are specified in the file `environment.yml`.
We use `conda` virtual environments to manage the project dependencies in
isolation.

Run the following command in the repository folder to create a separate environment 
and install all required dependencies in it:

```bash
$ conda env create -f environment.yml
```

## Reproducing the results

The [`bash`](bash) folder contains some utility bash scripts useful to run multiple experiments sequentially. They cover the training and evaluation pipeline of all the models tested in the paper. To let everything work as expected, please run them from the parent directory.

### Training

Please check out your disk size, these scripts will save two model checkpoints (best and the last one) for every seed.

Train **BERT** on the Misogyny (EN) dataset:

```bash
./bash/train_model_10_seeds.sh bert-base-uncased <output_dir> <training_dataset>
```

e.g., `./bash/train_model_10_seeds.sh bert-base-uncased . miso`


Train **BERT+EAR** on the Multilingual and Multi-Aspect dataset:

```bash
./bash/train_model_EAR_10_seeds.sh bert-base-uncased <output_dir> <training_dataset>
```

e.g., `./bash/train_model_EAR_10_seeds.sh bert-base-uncased . mlma`


Note that:
- if you want to take into account class imbalance, you should add the `--balanced_loss` to the parameters passed as command line arguments to python;
- for BERT+SOC (Kennedy et al. 2020), we re-use the authors's implementation. Therefore, no
training scripts are provided here.

## Testing

To evaluate a model, or a folder with several models (different seeds), you have to:
1. run the evaluation on synthetic data.
2. run the evaluation on test data 

### Evaluation of bias metrics on synthetic data

Here we provide an example to run the evaluation on Madlibs77K synthetic data using a specific checkpoint name (`last.ckpt` in this case).

```bash
./bash/evaluate_folder_madlibs_pattern.sh <in_dir> <out_dir> last.ckpt
```

Analogous script for the other synthetic sets are stored in the folder `./bash`. Namely:
- `evaluate_folder_miso_synt.sh` Run the evaluation of all the models within a specified parent directory on Misogyny (EN), synthetic set.
- `evaluate_folder_miso-ita_synt.sh` Run the evaluation of all the models within a specified parent directory on Misogyny (IT), synthetic set.

### Evaluation on test data

Here we provide an example to run the evaluation on the test set of MlMA.

```bash
./bash/test_folder.sh <in_dir> <out_dir> mlma <src_tokenizer> <ckpt_pattern>
```

Note that evaluation on Misogyny (IT) requires the parameter `--src_tokenizer dbmdz/bert-base-italian-uncased`

### Automatic Biased Term Extraction

We provide a Jupyter Notebook where we show how to extract terms with the lowest contextualization, which
may induce most of the bias in the model.

After having trained at least one model (i.e., you have a model checkpoint), the notebook [`term_extraction.ipynb`](term_extraction.ipynb) will guide you through the discovery of biased terms.

## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is not open source. The authors reserve the rights to the
article content, which is currently submitted for publication.
