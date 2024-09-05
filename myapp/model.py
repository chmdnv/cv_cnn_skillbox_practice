import os
import pickle
import logging
import torch
from torch import nn
from torchvision.transforms import ToTensor, Compose, Normalize, ToPILImage

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        
        self.model = nn.Sequential(
            self.layer1,
            self.layer2,
            nn.Flatten(),
            nn.Linear(in_features=2304, out_features=128),
            nn.Dropout(0.25),
            nn.Linear(in_features=128, out_features=n_classes)
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        predict = self.model(x)
        return predict.squeeze(1).argmax(dim=1, keepdim=True)


class Model:
    def __init__(self):
        self.meta = self.__class__.load_meta()
        self.char_map = self.meta['char map dict']
        self.model = self.load_model()

    def __repr__(self):
        return repr(self.model)

    def __str__(self):
        return self.model.__class__.__name__

    @staticmethod
    def load_meta():
        meta_path = os.path.join('..', 'models', 'model.pkl')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"{meta_path} does not exist!")
        with open(meta_path, 'rb') as file:
            meta = pickle.load(file)

        return meta

    def load_model(self):
        model_path = os.path.join('..', self.meta['model path'])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} does not exist!")

        model = CNN(len(self.char_map))
        model.load_state_dict(torch.load(model_path, weights_only=True))

        return model

    @staticmethod
    def prepare(x: torch.Tensor) -> torch.Tensor:
        transform = Compose([
            ToPILImage(),
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])

        x = transform(x.float())

        if not isinstance(x, torch.Tensor):
            raise TypeError('Input data type must be torch.Tensor')

        if x.shape == torch.Size([28, 28]):
            x = x.unsqueeze(0)

        if x.shape == torch.Size([1, 28, 28]):
            x = x.unsqueeze(0)

        return x

    def predict(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred_char : str
            Символ-предсказание
        """
        logger.info(f"Model: {self}, accuracy: {self.meta['accuracy'] :.2f}")

        x = self.__class__.prepare(x)

        logger.info(f"Starting prediction on {self} with shape {x.shape}")

        prediction = int(self.model.predict(x))
        logger.info(f"Predicted class: {prediction}")

        if prediction not in self.char_map:
            logger.warning(f"{repr(prediction)} not found in char map dict!")

        pred_char = self.char_map.get(prediction, None)
        logger.info(f"Predicted character: {pred_char}")
            
        return pred_char


# Test
if __name__ == '__main__':
    x = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 33, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 244, 112, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 246, 49, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 246, 49, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 233, 21, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 202, 217, 4, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 217, 203, 4, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 187, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 59, 245, 46, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 238, 222, 115, 1, 0, 0, 0, 0, 0, 0, 2, 214, 245, 4, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 159, 245, 142, 220, 175, 84, 5, 0, 0, 0, 2, 127, 251, 95, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 244, 95, 0, 22, 172, 231, 100, 5, 0, 0, 110, 235, 218, 20, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 144, 91, 1, 0, 0, 0, 7, 190, 202, 22, 3, 220, 221, 9, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 4, 205, 172, 1, 0, 0, 0, 0, 0, 6, 123, 219, 174, 250, 83, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 51, 236, 121, 3, 0, 0, 0, 0, 0, 0, 0, 0, 191, 254, 127, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 3, 218, 214, 2, 0, 0, 0, 0, 0, 0, 0, 0, 24, 251, 175, 7, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 15, 201, 206, 12, 0, 0, 0, 0, 0, 0, 0, 0, 186, 246, 159, 4, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 37, 215, 24, 0, 0, 0, 0, 0, 0, 0, 0, 10, 246, 245, 22, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 65, 97, 11, 0, 0, 0, 0, 0, 0, 0, 0, 2, 128, 246, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 7, 154, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 235, 234, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 8, 142, 123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 215, 255, 187, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 22, 218, 91, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 185, 172, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 100, 228, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                     ).t().float()
    model = Model()
    pred = model.predict(x)
    print(pred)
