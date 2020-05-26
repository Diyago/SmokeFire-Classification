import codecs
import warnings

import pytorch_lightning as pl
import torch
from poyo import parse_string
from torch.utils.data import DataLoader

from common_blocks.datasets import TrainDataset
from common_blocks.logger import init_logger
from common_blocks.transforms import get_transforms
from common_blocks.utils import seed_torch, create_folds
from models.lightningclassifier import LightningClassifier

with codecs.open("config/config_classification.yml", encoding="utf-8") as ymlfile:
    config_yaml = ymlfile.read()
    config = parse_string(config_yaml)

LOGGER = init_logger(config['logger_path']['main_logger'])
warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == '__main__':
    # TODO inference for custom dataset
    #
    seed_torch(seed=config['total_seed'])
    folds = create_folds(config['validation'])
    for fold in range(config['validation']['nfolds']):
        trn_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index

        valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True),
                                     config['Val']['Dataset'],
                                     transform=get_transforms(data='valid'))
        valid_loader = DataLoader(valid_dataset, **config['Val']['loader'])

        model = LightningClassifier(config)
        checkpoint = torch.load('./lightning_logs/efficientnet_b2b/fold_1/epoch=06-avg_val_metric=0.9976.ckpt',
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        trainer = pl.Trainer()
        # todo fix returning results
        results = trainer.test(model, test_dataloaders=valid_loader)
        break
