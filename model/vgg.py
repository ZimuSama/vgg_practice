from torch import nn
import collections

VGGconf = {
        'A':collections.OrderedDict([
          ('layer1_conv1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
          ('layer1_relu1', nn.ReLU()),
          ('layer1_pool', nn.MaxPool2d(2)),
          ('layer2_conv1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
          ('layer2_relu1', nn.ReLU()),
          ('layer2_pool', nn.MaxPool2d(2)),
          ('layer3_conv1', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
          ('layer3_relu1', nn.ReLU()),
          ('layer3_conv2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
          ('layer3_relu2', nn.ReLU()),
          ('layer3_pool', nn.MaxPool2d(2)),
          ('layer4_conv1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
          ('layer4_relu1', nn.ReLU()),
          ('layer4_conv2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer4_relu2', nn.ReLU()),
          ('layer4_pool', nn.MaxPool2d(2)),
          ('layer5_conv1', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer5_relu1', nn.ReLU()),
          ('layer5_conv2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer5_relu2', nn.ReLU()),
          ('layer5_pool', nn.MaxPool2d(2))
        ]),
        'B':collections.OrderedDict([
          ('layer1_conv1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
          ('layer1_relu1', nn.ReLU()),
          ('layer1_conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
          ('layer1_relu2', nn.ReLU()),
          ('layer1_pool', nn.MaxPool2d(2)),
          ('layer2_conv1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
          ('layer2_relu1', nn.ReLU()),
          ('layer2_conv2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
          ('layer2_relu2', nn.ReLU()),
          ('layer2_pool', nn.MaxPool2d(2)),
          ('layer3_conv1', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
          ('layer3_relu1', nn.ReLU()),
          ('layer3_conv2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
          ('layer3_relu2', nn.ReLU()),
          ('layer3_pool', nn.MaxPool2d(2)),
          ('layer4_conv1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
          ('layer4_relu1', nn.ReLU()),
          ('layer4_conv2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer4_relu2', nn.ReLU()),
          ('layer4_pool', nn.MaxPool2d(2)),
          ('layer5_conv1', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer5_relu1', nn.ReLU()),
          ('layer5_conv2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer5_relu2', nn.ReLU()),
          ('layer5_pool', nn.MaxPool2d(2))
        ]),
        'C':collections.OrderedDict([
          ('layer1_conv1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
          ('layer1_relu1', nn.ReLU()),
          ('layer1_conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
          ('layer1_relu2', nn.ReLU()),
          ('layer1_pool', nn.MaxPool2d(2)),
          ('layer2_conv1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
          ('layer2_relu1', nn.ReLU()),
          ('layer2_conv2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
          ('layer2_relu2', nn.ReLU()),
          ('layer2_pool', nn.MaxPool2d(2)),
          ('layer3_conv1', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
          ('layer3_relu1', nn.ReLU()),
          ('layer3_conv2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
          ('layer3_relu2', nn.ReLU()),
          ('layer3_conv3', nn.Conv2d(256, 256, kernel_size=1)),
          ('layer3_relu3', nn.ReLU()),
          ('layer3_pool', nn.MaxPool2d(2)),
          ('layer4_conv1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
          ('layer4_relu1', nn.ReLU()),
          ('layer4_conv2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer4_relu2', nn.ReLU()),
          ('layer4_conv3', nn.Conv2d(512, 512, kernel_size=1)),
          ('layer4_relu3', nn.ReLU()),
          ('layer4_pool', nn.MaxPool2d(2)),
          ('layer5_conv1', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer5_relu1', nn.ReLU()),
          ('layer5_conv2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer5_relu2', nn.ReLU()),
          ('layer5_conv3', nn.Conv2d(512, 512, kernel_size=1)),
          ('layer5_relu3', nn.ReLU()),
          ('layer5_pool', nn.MaxPool2d(2))
        ]),
        'D':collections.OrderedDict([
          ('layer1_conv1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
          ('layer1_relu1', nn.ReLU()),
          ('layer1_conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
          ('layer1_relu2', nn.ReLU()),
          ('layer1_pool', nn.MaxPool2d(2)),
          ('layer2_conv1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
          ('layer2_relu1', nn.ReLU()),
          ('layer2_conv2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
          ('layer2_relu2', nn.ReLU()),
          ('layer2_pool', nn.MaxPool2d(2)),
          ('layer3_conv1', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
          ('layer3_relu1', nn.ReLU()),
          ('layer3_conv2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
          ('layer3_relu2', nn.ReLU()),
          ('layer3_conv3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
          ('layer3_relu3', nn.ReLU()),
          ('layer3_pool', nn.MaxPool2d(2)),
          ('layer4_conv1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
          ('layer4_relu1', nn.ReLU()),
          ('layer4_conv2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer4_relu2', nn.ReLU()),
          ('layer4_conv3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer4_relu3', nn.ReLU()),
          ('layer4_pool', nn.MaxPool2d(2)),
          ('layer5_conv1', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer5_relu1', nn.ReLU()),
          ('layer5_conv2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer5_relu2', nn.ReLU()),
          ('layer5_conv3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer5_relu3', nn.ReLU()),
          ('layer5_pool', nn.MaxPool2d(2))
        ]),
        'E':collections.OrderedDict([
          ('layer1_conv1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
          ('layer1_relu1', nn.ReLU()),
          ('layer1_conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
          ('layer1_relu2', nn.ReLU()),
          ('layer1_pool', nn.MaxPool2d(2)),
          ('layer2_conv1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
          ('layer2_relu1', nn.ReLU()),
          ('layer2_conv2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
          ('layer2_relu2', nn.ReLU()),
          ('layer2_pool', nn.MaxPool2d(2)),
          ('layer3_conv1', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
          ('layer3_relu1', nn.ReLU()),
          ('layer3_conv2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
          ('layer3_relu2', nn.ReLU()),
          ('layer3_conv3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
          ('layer3_relu3', nn.ReLU()),
          ('layer3_conv4', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
          ('layer3_relu4', nn.ReLU()),
          ('layer3_pool', nn.MaxPool2d(2)),
          ('layer4_conv1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
          ('layer4_relu1', nn.ReLU()),
          ('layer4_conv2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer4_relu2', nn.ReLU()),
          ('layer4_conv3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer4_relu3', nn.ReLU()),
          ('layer4_conv4', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer4_relu4', nn.ReLU()),
          ('layer4_pool', nn.MaxPool2d(2)),
          ('layer5_conv1', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer5_relu1', nn.ReLU()),
          ('layer5_conv2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer5_relu2', nn.ReLU()),
          ('layer5_conv3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer5_relu3', nn.ReLU()),
          ('layer5_conv4', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
          ('layer5_relu4', nn.ReLU()),
          ('layer5_pool', nn.MaxPool2d(2))
        ])
}

class NeuralNetwork(nn.Module):
    def __init__(self, type, classes):
        super().__init__()
        self.conv = nn.Sequential(VGGconf[type])
        self.dense = nn.Sequential(
            nn.Flatten(1,3),
            nn.Linear(512*7*7, 4096), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, 0, 1e-2)
                # nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                m.bias.data.fill_(0.)
            elif isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.)

    def forward(self, x):
        a = self.conv(x)
        a = self.dense(a)
        return a