import torch
from torch import nn

import common

DEBUG = False # True: 1k image overfit test results don't get saved, False: full run
EPOCHS = 15
"""
IDEA:
image (3x32x32) → cut into 64 patches of 4x4 → each patch becomes a 128-number "word"
→ add a [CLS] word at the front → add position info → 6 transformer layers
→ read the [CLS] word → 100 class scores
"""

class ViT(nn.Module):
    def __init__(self, patch=4, dim=128, depth=6, heads=4, mlp_dim=256,
                 num_classes=common.NUM_CLASSES):
        super().__init__()
        n_patches = (32 // patch) ** 2  # 8 x 8 = 64 patches

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1,1,dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)

        layer = nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout=0.1, activation="gelu",
                                      batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)


    def forward(self, x ):
        x = self.patch_embed(x).flatten(2).transpose(1, 2) # (B, 128, 8, 8) -> (B, 64, 128)
        cls = self.cls_token.expand(len(x), -1, -1)        # (1, 1, 128)    -> (B, 1, 128)
        x = torch.cat([cls, x], dim=1) + self.pos_embed    #                -> (B, 65, 128)
        x = self.encoder(x)                                #                -> (B, 65, 128)
        return self.head(self.norm(x[:, 0]))               # CLS only: (B, 128) -> (B, 100)


def main():
    common.set_seed(common.SEED)
    device = common.get_device()
    train_loader, val_loader, test_loader = common.get_loaders(debug_size=1000 if DEBUG else None)

    model = ViT().to(device)
    print("params:", common.count_param(model))
    optimizer = torch.optim.Adam(model.parameters(), lr =5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    train_time = common.fit(model, train_loader, val_loader, optimizer, epochs=EPOCHS,
                            device=device, name="vit", scheduler=scheduler)
    y_true, y_pred, probs, guess_time = common.timed_predict(model, test_loader, device)
    if not DEBUG:
        common.save_results("vit", y_true, y_pred, probs, train_time, guess_time, common.count_param(model))

    metrics = common.compute_metrics(y_true, y_pred, probs)
    print("test |", " | ".join(f"{name}  {value:.3f}" for name, value in metrics.items()))
    print(f"train {train_time:.0f}s ({train_time / EPOCHS:.1f}s per epoch), inference {guess_time:.2f}s")



if __name__ == "__main__":
    main()