# dnn-final

Image classification on CIFAR-100: baseline models vs. transformer-based architectures.

## Models

**Baseline evaluation**
- `dummy_classifier.py`: Dummy classifier (chance-level floor)
- `logistic_regression.py`: Logistic regression on flattened pixels
- `cnn.py`: 2D-CNN

**Proposed architectures**
- `vit.py`: Vision Transformer
- `cnn_transformer.py`: 2D-CNN + Transformer (multi-head attention)

## Project structure

```
dnn-final/
├── README.md
├── dummy_classifier.py
├── logistic_regression.py
├── cnn.py
├── vit.py
├── cnn_transformer.py
├── results.ipynb
└── dataset/
```

## What we report (`results.ipynb`)
1. **Performance metrics:** accuracy, F1-score (macro), etc.
2. **Training and inference time.**
3. **Model diagnostics:** confusion matrix, examples of misclassified images.
4. **Comparison with previous studies.**

## Resources

### 1. Data loading
- [torchvision `CIFAR100`](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html)
- [Transforms v2 overview](https://docs.pytorch.org/vision/stable/transforms.html)
- [CutMix and MixUp example](https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_cutmix_mixup.html)

### 2. Non-neural baselines (`dummy_classifier.py`, `logistic_regression.py`)
- [scikit-learn `DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
- [scikit-learn `LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn `PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) (optional: reduces 3072 pixel features before logistic regression)

### 3. 2D-CNN baseline (`cnn.py`)
- [PyTorch "Training a Classifier" tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Optimization loop basics](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [d2l.ai: Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html)
- [d2l.ai: Modern CNNs (ResNet)](https://d2l.ai/chapter_convolutional-modern/resnet.html)

### 4. Transformer fundamentals and ViT (`vit.py`)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [d2l.ai: Attention and Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)
- [d2l.ai: Vision Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html)
- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)

### 5. 2D-CNN + Transformer (`cnn_transformer.py`)
- [CCT: Escaping the Big Data Paradigm with Compact Transformers](https://arxiv.org/abs/2104.05704)
- [CCT official repo](https://github.com/SHI-Labs/Compact-Transformers)
- [DHVT (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/file/5e0b46975d1bfe6030b1687b0ada1b85-Paper-Conference.pdf)
- [`nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [`nn.TransformerEncoderLayer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)

### 6. Training recipe
- [`AdamW`](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [`LinearLR`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html), [`CosineAnnealingLR`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html), [`SequentialLR`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html)
- [`CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) (`label_smoothing`)
- [MPS backend (Mac GPU)](https://docs.pytorch.org/docs/stable/notes/mps.html)

### 7. Metrics and diagnostics
- [scikit-learn model evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [`f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) (use `average="macro"` for 100 classes)
- [`classification_report`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [`confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) and [`ConfusionMatrixDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)

### 8. Training and inference time
- [`time.perf_counter`](https://docs.python.org/3/library/time.html#time.perf_counter)
- [PyTorch benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [`torch.mps.synchronize`](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html) (call before reading the timer, GPU work is asynchronous)

### 9. Previous studies (for comparison)
- [ViT (Dosovitskiy et al., ICLR 2021)](https://arxiv.org/abs/2010.11929)
- [DeiT (Touvron et al., ICML 2021)](https://arxiv.org/abs/2012.12877)
- [Vision Transformer for Small-Size Datasets (Lee et al., 2021)](https://arxiv.org/abs/2112.13492)
- [Efficient Training of Visual Transformers with Small Datasets (Liu et al., NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/file/c81e155d85dae5430a8cee6f2242e82c-Paper.pdf)
- [How to Train Vision Transformer on Small-scale Datasets? (Gani et al., BMVC 2022)](https://arxiv.org/abs/2210.07240)
- [CCT (Hassani et al., 2021)](https://arxiv.org/abs/2104.05704)
- [DHVT (Lu et al., NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/file/5e0b46975d1bfe6030b1687b0ada1b85-Paper-Conference.pdf)
- [MSCViT (Zhang & Zhang, 2025)](https://arxiv.org/abs/2501.06040)
- [Full sortable paper list](https://claude.ai/artifact/UqZm2keXnWmapeNBLvasv5)

## Plan
1. Dummy classifier and logistic regression to set the floor.
2. 2D-CNN baseline to ~60–70% on CIFAR-100.
3. Vision Transformer (patch size 4) trained from scratch.
4. 2D-CNN + Transformer: swap the ViT patch embedding for a conv tokenizer.
5. Report metrics, timing, diagnostics and comparison with previous studies in `results.ipynb`.

