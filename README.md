# Pytorch-Lightning-practice
For OpenSource-Contribution-Academy

본 Repo는 2022년 오픈소스 컨트리뷰톤의 Pytorch-Lightning 프로젝트 신청을 위하여 작성되었습니다.  

### Installation

본 프로젝트에 사용된 ```Python```의 버전은 ```3.8.13```입니다.  
버전에 맞는 Python 설치 후,  
```pip install -r requirements.txt```  
명령을 이용하여, dependency를 설치해주시면 됩니다.

### LightningModule 클래스 
```python
class MNISTModel(pl.LightningModule):
    def __init__(self, data_dir,batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor()])

        self.network = nn.Sequential(
            nn.Conv2d(1, 8, 4, stride=2, bias=False),  
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=2, bias=False), 
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, bias=False), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*4*4,10)
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
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
    
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
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

```

### Usage

```python main.py```

