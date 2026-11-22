import torch
from torch import nn

import common

DEBUG = False # True: 1k image overfit test, False: full run
EPOCHS = 15

class ResidualBlock(nn.Module): # 2 convolutions plus a shortcut that skips over them
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        #shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))  # the residual connection

class ResNet(nn.Module):
    def __init__(self, widths=(32, 64, 128), num_classes=common.NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(),
        )

        # 3 stages of 2 blocks, each stage halves the img size and doubles the channels
        layers, in_ch = [], widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            layers += [ResidualBlock(in_ch, w, stride), ResidualBlock(w, w)]
            in_ch = w
        self.stages = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1) # averaging each each channel down to one number
        self.head = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)             #(B, 3, 32, 32) -> (B, 32, 32, 32)
        x = self.stages(x)           #               -> (B, 128, 8, 8)
        x = self.pool(x).flatten(1)  #               -> (B, 128)
        return self.head(x)          #               -> (B, 100)


def main():
    common.set_seed(common.SEED)
    device = common.get_device()
    train_loader, val_loader, test_loader = common.get_loaders(debug_size=1000 if DEBUG else None)

    model = ResNet().to(device)
    print("params:", common.count_param(model))
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    train_time = common.fit(model, train_loader, val_loader, optimizer,
                            epochs=EPOCHS, device=device, name='cnn', scheduler=scheduler)
    y_true, y_pred, probs, guess_time = common.timed_predict(model, test_loader, device)
    if not DEBUG:  # never save debug number as real results
        common.save_results("cnn", y_true, y_pred, probs,
                            train_time, guess_time, common.count_param(model))

    metrics = common.compute_metrics(y_true, y_pred, probs)
    print("test |", " | ".join(f"{name} {value:.3f}" for name, value in metrics.items()))
    print(f"train {train_time:.0f}s ({train_time / EPOCHS:.1f}s per epoch), inference {guess_time:.2f}s")

if __name__ == "__main__":
    main()
