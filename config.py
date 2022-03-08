"""Main module."""

# Define here main default values

DEFAULT_OUT_DIR = "./runs"
DEFAULT_EMBEDDINGS_PATH = "./data/glove.6B/glove.6B.100d.txt"
DEFAULT_SAVE_DIR = "./dumps"
DEFAULT_OUT_DIR = "./runs"
DEFAULT_HPARAMS_CNN = {
    "max_sequence_length": 120,  #  original 250
    "max_num_words": 32000,  #  original 10000
    "embedding_dim": 100,
    "embedding_trainable": False,
    "learning_rate": 0.00005,
    "stop_early": True,
    "es_patience": 5,  # Only relevant if STOP_EARLY = True, original: 1
    "es_min_delta": 0,  # Only relevant if STOP_EARLY = True
    "batch_size": 64,  #  original 128
    "epochs": 30,  #  original 20
    "dropout_rate": 0.3,
    "cnn_filter_sizes": [128, 128, 128],
    "cnn_kernel_sizes": [5, 5, 5],
    "cnn_pooling_sizes": [5, 5, 40],
    "verbose": True,
}
