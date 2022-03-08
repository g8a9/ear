"""Collect and expose datasets for experiments."""
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
import pandas as pd
from operator import itemgetter
import logging
import os


logging.basicConfig(
    format="%(levelname)s:%(asctime)s:%(module)s:%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


MADLIBS_DATASETS = ["madlibs77k", "madlibs89k"]
TOX_DATASETS = ["tox_nonfuzz", "tox_fuzz"]

MISO_DATASETS = ["miso", "miso-ita-raw", "miso-ita-synt"]
MISOSYNT_DATASETS = ["miso_synt_test"]

MLMA_DATASETS = ["mlma"]
MLMA_RAW_DATASETS = ["mlma_en", "mlma_fr", "mlma_ar"]


AVAIL_DATASETS = (
    MADLIBS_DATASETS
    + TOX_DATASETS
    + MISO_DATASETS
    + MISOSYNT_DATASETS
    + MLMA_DATASETS
)


def get_dataset_by_name(name: str, base_dir=None):
    path = os.path.join(base_dir, name) if base_dir else name
    
    train, dev, test = None, None, None
    if name in MADLIBS_DATASETS:
        test = Madlibs.build_dataset(path)
    elif name in TOX_DATASETS:
        test = Toxicity.build_dataset(path)
    elif name in MISO_DATASETS:
        if name == "miso-ita-synt":
            test = MisoDataset.build_dataset(name, "test")
        else:
            train = MisoDataset.build_dataset(name, "train")
            dev = MisoDataset.build_dataset(name, "dev")
            test = MisoDataset.build_dataset(name, "test")
    elif name in MISOSYNT_DATASETS:
        test = MisoSyntDataset.build_dataset(name)
    elif name in MLMA_RAW_DATASETS:
        test = MLMARawDataset.build_dataset(name)
    elif name in MLMA_DATASETS:
        train = MLMADataset.build_dataset(split="train")
        dev = MLMADataset.build_dataset(split="dev")
        test = MLMADataset.build_dataset(split="test")
    else:
        raise ValueError(f"Can't recognize dataset name {name}")
    return train, dev, test


def get_tokenized_path(path: str):
    base_dir, filename = os.path.dirname(path), os.path.basename(path)
    return os.path.join(base_dir, f"{os.path.splitext(filename)[0]}.pt")


class MLMARawDataset(Dataset):
    #  DEPRECATED
    """Multilingual and Multi-Aspect Hate Speech Analysis"""

    def __init__(self, path: str):
        self.path = path
        data = pd.read_csv(path)

        # define the hate binary label
        data["hate"] = 1
        data.loc[data.sentiment == "normal", "hate"] = 0
        data = data.loc[
            data.sentiment.apply(lambda x: "normal" not in x or x == "normal")
        ]

        self.data = data
        self.texts = data["tweet"].tolist()
        self.labels = data["hate"].astype(int).tolist()
        self.tokenized_path = get_tokenized_path(path)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels

    @classmethod
    def build_dataset(cls, name: str):
        if name == "mlma_en":
            return cls(os.path.join("data", "hate_speech_mlma", f"en_dataset.csv"))
        elif name == "mlma_fr":
            return cls(os.path.join("data", "hate_speech_mlma", f"fr_dataset.csv"))
        elif name == "mlma_ar":
            return cls(os.path.join("data", "hate_speech_mlma", f"ar_dataset.csv"))
        else:
            raise ValueError("Name not recognized.")


class MLMADataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        data = pd.read_csv(path, sep="\t")
        self.texts = data["tweet"].tolist()
        self.labels = data["hate"].astype(int).tolist()
        self.tokenized_path = get_tokenized_path(path)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels

    @classmethod
    def build_dataset(cls, split: str):
        return cls(f"./data/mlma_{split}.tsv")


class MisoDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        data = pd.read_csv(path, sep="\t")
        self.texts = data["text"].tolist()
        self.labels = data["misogynous"].astype(int).tolist()
        self.tokenized_path = get_tokenized_path(path)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels

    @classmethod
    def build_dataset(cls, name: str, split: str):

        if name == "miso":
            return cls(f"./data/miso_{split}.tsv")
        elif name == "miso-ita-raw":
            return cls(f"./data/AMI2020_{split}_raw.tsv")
        elif name == "miso-ita-synt":
            return cls(f"./data/AMI2020_{split}_synt.tsv")
        else:
            raise ValueError("Type not recognized.")


class MisoSyntDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        data = pd.read_csv(path, sep="\t", header=None, names=["Text", "Label"])
        self.texts = data["Text"].tolist()
        self.labels = data["Label"].astype(int).tolist()
        self.tokenized_path = get_tokenized_path(path)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels

    @classmethod
    def build_dataset(cls, type: str):
        if type not in MISOSYNT_DATASETS:
            raise ValueError("Type not recognized.")
        else:
            return cls(f"./data/miso_synt_test.tsv")


class Madlibs(Dataset):
    def __init__(self, path: str):
        self.path = path
        data = pd.read_csv(path)
        #  Use the same convention for binary labels: 0 (NOT_BAD/FALSE), 1 (BAD/TRUE)
        self.texts = data["Text"].tolist()
        self.labels = pd.get_dummies(data.Label)["BAD"].tolist()
        self.tokenized_path = get_tokenized_path(path)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels

    @classmethod
    def build_dataset(cls, type: str):
        if type not in MADLIBS_DATASETS:
            raise ValueError("Type not recognized.")
        if type == "madlibs77k":
            return cls(f"./data/bias_madlibs_77k.csv")
        else:
            return cls(f"./data/bias_madlibs_89k.csv")


class TokenizerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        batch_size,
        max_seq_length,
        num_workers,
        pin_memory,
        load_pre_tokenized=False,
        store_pre_tokenized=False,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.load_pre_tokenized = load_pre_tokenized
        self.store_pre_tokenized = store_pre_tokenized

        self.train, self.val, self.test = get_dataset_by_name(dataset_name)
        self.train_steps = int(len(self.train) / batch_size)

    def prepare_data(self):
        train, val, test = self.train, self.val, self.test

        for split in [train, val, test]:
            if self.load_pre_tokenized and os.path.exists(split.tokenized_path):
                logging.info(
                    """
                    Loading pre-tokenized dataset.
                    Beware! Using pre-tokenized embeddings could not match you choice for max_length
                    """
                )
                continue

            if self.load_pre_tokenized:
                logging.info(f"Load tokenized but {split.tokenized_path} is not found")

            logger.info("Tokenizing...")
            encodings = self.tokenizer(
                split.get_texts(),
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="pt",
            )

            if self.store_pre_tokenized:
                logger.info(f"Saving to {split.tokenized_path}")
                torch.save(encodings, split.tokenized_path)

    def setup(self, stage=None):
        if stage == "fit":
            train, val = self.train, self.val

            logging.info(f"TRAIN len: {len(train)}")
            logging.info(f"VAL len: {len(val)}")

            train_encodings = torch.load(train.tokenized_path)
            train_labels = torch.LongTensor([r["label"] for r in train])
            self.train_data = EncodedDataset(train_encodings, train_labels)

            val_encodings = torch.load(val.tokenized_path)
            val_labels = torch.LongTensor([r["label"] for r in val])
            self.val_data = EncodedDataset(val_encodings, val_labels)

        elif stage == "test":
            test = self.test
            logging.info(f"TEST len: {len(test)}")

            test_encodings = torch.load(test.tokenized_path)
            test_labels = torch.LongTensor([r["label"] for r in test])
            self.test_data = EncodedDataset(test_encodings, test_labels)

        else:
            raise ValueError(f"Stage {stage} not known")

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class EncodedDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return self.labels.shape[0]


class PlainDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        return {"text": self.texts[index], "label": self.labels[index]}

    def __len__(self):
        return len(self.labels)
