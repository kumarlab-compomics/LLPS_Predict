##########
# Script for testing sequence-structure model (ESM2 & SaProt)
#
# Usage: 
# cd LLPS_Predict
# python -m src.scripts.predict_llps_score [--config c] [--model_name m] 
#   [--data_path d] [--score_path s] [--batch_size n] [--dropout p] [threshold t]
#
# Override config parameters with command line arguments if provided
##########

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.loaders.dataset_pair import LLPSDatasetPair
from src.models.model_ensemble import EnsembleClassifier
# evaluation metrics
from src.utils.metrics import classification_metrics
# config parameters
import yaml
import argparse
# predict loop
from src.utils.predictor import predict
# results
import pandas as pd


# default parameters
DEFAULTS_DICT = {
    'ensemble': {
        'embed_model1': 'esm2_650m', 
        'embed_model2': 'saprot_650m', 
        'embedding_dim': 1280, 
        'num_layers': 1, 
        'num_heads': 4, 
        'dim_feedforward': 1280, 
        'dropout': 0.6,
        'checkpoint_path': 'checkpoints/trained_models/ensemble_baseline_h4_l1.pth'
    },
    'ensemble_driver': {
        'embed_model1': 'esm2_650m', 
        'embed_model2': 'saprot_650m', 
        'embedding_dim': 1280, 
        'num_layers': 1, 
        'num_heads': 4, 
        'dim_feedforward': 1280, 
        'dropout': 0.6,
        'checkpoint_path': 'checkpoints/trained_models/ensemble_driver_baseline_h4_l1.pth'
    },
    'ensemble_partner': {
        'embed_model1': 'esm2_650m', 
        'embed_model2': 'saprot_650m', 
        'embedding_dim': 1280, 
        'num_layers': 1, 
        'num_heads': 4, 
        'dim_feedforward': 1280, 
        'dropout': 0.6,
        'checkpoint_path': 'checkpoints/trained_models/ensemble_partner_baseline_h4_l1.pth'
    }
}

# command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument(
    '--model_name', 
    type=str, 
    choices=["ensemble", "ensemble_driver", "ensemble_partner"], 
    required=True
)
parser.add_argument('--data_path', type=str)
parser.add_argument('--score_path', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--dropout', type=float)
parser.add_argument('--threshold', type=float)
args = parser.parse_args()

config = dict(DEFAULTS_DICT[args.model_name])
# override with config parameters if provided
if args.config is not None:
    with open(args.config, 'r') as f:
        loaded = yaml.safe_load(f) or {}
    config.update(loaded)
# override with command line arguments if provided
args_dict = {
    k: v for k, v in vars(args).items()
    if v is not None and k != "config"
}
config.update(args_dict)

# datasets
TEST_DATA_PATH = config['data_path']
SQN_COLUMN1 = config['sqn_column1']
SQN_COLUMN2 = config['sqn_column2']
LABEL_COLUMN = config['label_column']
SCORE_PATH = config['score_path']

# model
SQN_EMBED_MODEL1 = config['sqn_embed_model1']
SQN_EMBED_MODEL2 = config['sqn_embed_model2']
SQN_EMBED_LOAD_LOCAL = config['sqn_embed_load_local']
SQN_EMBED_DIM = config['sqn_embed_dim']
N_LAYER = config['n_layer']
N_HEAD = config['n_head']
FEEDFORWARD_DIM = config['feedforward_dim']
CHECKPOINT_PATH = config['checkpoint_path']
THRESHOLD = config['threshold']

# configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = config['batch_size']
DROPOUT = config['dropout']

print('Data path:', TEST_DATA_PATH)

# Dataloader
dataset_test = LLPSDatasetPair(
    TEST_DATA_PATH, SQN_COLUMN1, SQN_COLUMN2, LABEL_COLUMN
)
dataloader_test = DataLoader(
    dataset_test, batch_size=BATCH_SIZE, shuffle=False
)

# Testing
# model
classifier = EnsembleClassifier(
    embed_model1=SQN_EMBED_MODEL1,
    embed_model2=SQN_EMBED_MODEL2,
    embedding_dim=SQN_EMBED_DIM,
    num_layers=N_LAYER,
    num_heads=N_HEAD,
    dim_feedforward=FEEDFORWARD_DIM,
    dropout=DROPOUT,
    local_files_only=SQN_EMBED_LOAD_LOCAL
)
classifier = classifier.to(DEVICE)

# checkpoint path
checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
classifier.load_state_dict(checkpoint['model'], strict=False)

# loss function
criterion = nn.BCEWithLogitsLoss()

# compute test loss
# get predictions
pred = predict(
    model=classifier, 
    dataloader=dataloader_test,
    criterion=criterion,
    device=DEVICE,
    verbose=True
)
sequence = dataset_test.sequences2
target = dataset_test.labels
df_pred = pd.DataFrame({
    'Sequence': sequence, 'Label': target, 'Prediction': pred
})
df_pred.to_csv(SCORE_PATH, index=False)


# Compute Metrics
classification_metrics(target, pred, threshold=THRESHOLD)
