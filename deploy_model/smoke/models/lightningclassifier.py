import pytorch_lightning as pl

from models.pretrained_models import get_model_output


class LightningClassifier(pl.LightningModule):
    def __init__(self, config):
        super(LightningClassifier, self).__init__()
        self.hparams = config
        self.model = get_model_output(**config['model_params'])  # CustomSEResNeXt(config['model_params'])
        self.val_metrics = []

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        return self.model.forward(x)
