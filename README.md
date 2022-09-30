# Pytorch-Lightning-practice

### Installation

본 프로젝트에 사용된 ```Python```의 버전은 ```3.8.13```입니다.  
버전에 맞는 Python 설치 후,  
```pip install -r requirements.txt```  
명령을 이용하여, dependency를 설치해주시면 됩니다.

### LightningModule 클래스 
```python
class CIFARModel(pl.LightningModule):
    def __init__(self, data_dir,batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor()])

        self.network = nn.Sequential(
        nn.Conv2d(3, 6, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10),
        )

        self.accuracy = Accuracy()

    def forward(self, x):
        out = self.network(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        h = self(x)
        loss = F.cross_entropy(h, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        h = self(x)
        loss = F.cross_entropy(h, y)
        preds = torch.argmax(h, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            CIFAR_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.CIFAR_train, self.CIFAR_val = random_split(CIFAR_full, [45000, 5000])

        if stage == "test" or stage is None:
            self.CIFAR_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.CIFAR_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.CIFAR_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.CIFAR_test, batch_size=self.batch_size)
```

### Usage

```python main.py```

