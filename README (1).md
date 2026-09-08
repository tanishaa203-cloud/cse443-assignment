# Problem Set 01 — Chest X-Ray Pneumonia Classification (CNN)

## Problem Statement
Build a Convolutional Neural Network (CNN) to classify pediatric chest X-ray images
(anterior-posterior view) into two categories: **NORMAL** and **PNEUMONIA**.
The dataset contains 5,863 JPEG images split into `train`, `test`, and `val` folders,
each with `NORMAL` and `PNEUMONIA` subfolders.

## Approach

1. **Data loading & preprocessing** — Images are loaded with Keras'
   `ImageDataGenerator`, resized to 150x150, converted to grayscale, and pixel
   values rescaled to the [0, 1] range.
2. **Data augmentation** — Applied to the training set only (rotation, width/height
   shift, shear, zoom, horizontal flip) to reduce overfitting and improve
   generalization, since the training set is relatively small for a deep CNN.
3. **Class imbalance handling** — The dataset has significantly more PNEUMONIA
   images than NORMAL images. Class weights were computed (`sklearn`'s
   `compute_class_weight`) and passed into training so the model doesn't bias
   toward the majority class.
4. **Model architecture** — A CNN with four convolutional blocks (Conv2D +
   BatchNormalization + MaxPooling), increasing filter sizes (32 → 64 → 128 → 128),
   followed by a dense layer (256 units) with dropout (0.5) to reduce overfitting,
   and a final sigmoid output for binary classification.
5. **Training** — Adam optimizer (lr=0.0001), binary cross-entropy loss, with
   `EarlyStopping` (to stop once validation loss stops improving) and
   `ReduceLROnPlateau` (to lower the learning rate when progress stalls).
6. **Evaluation** — Accuracy, precision, recall, a full classification report, and
   a confusion matrix on the held-out test set.

## Repository Structure

```
problem_set_01/
├── chest_xray_cnn.ipynb   # Full notebook: data loading -> training -> evaluation
├── training_history.png   # Accuracy/loss curves (generated after running)
├── confusion_matrix.png   # Test set confusion matrix (generated after running)
└── README.md               # This file
```

## How to Run
1. Open `chest_xray_cnn.ipynb` in Google Colab.
2. Set Runtime → Change runtime type → GPU.
3. Upload/mount the dataset zip via Google Drive (see the first cell) and update
   `ZIP_PATH` to its location.
4. Run all cells top to bottom.

## Findings

**Test set results (624 images):**

| Metric | Value |
|---|---|
| Test Accuracy | 0.9087 |
| Test Precision | 0.9071 |
| Test Recall | 0.9513 |
| Test Loss | 0.2549 |

**Classification report:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| NORMAL | 0.91 | 0.84 | 0.87 | 234 |
| PNEUMONIA | 0.91 | 0.95 | 0.93 | 390 |
| **Accuracy** | | | **0.91** | 624 |
| Macro avg | 0.91 | 0.89 | 0.90 | 624 |
| Weighted avg | 0.91 | 0.91 | 0.91 | 624 |

**Confusion Matrix:** see `confusion_matrix.png`
**Training/Validation curves:** see `training_history.png`

**Observations:**
- The model achieves ~91% overall accuracy on unseen test data, which is a
  strong result for a CNN trained from scratch (no transfer learning) on a
  dataset of this size.
- **Recall on PNEUMONIA (0.95) is notably higher than recall on NORMAL (0.84).**
  This is the most clinically relevant outcome: in a screening context, a false
  negative (telling a sick patient they're healthy) is far more costly than a
  false positive (flagging a healthy patient for further review). The model is
  correctly biased toward catching pneumonia cases, which is the desired
  behavior for this kind of triage tool.
- The trade-off for that high pneumonia recall is a somewhat lower recall on
  NORMAL cases (0.84) — the model produces some false positives, over-flagging
  a portion of healthy X-rays as pneumonia. This is an acceptable, and often
  intentional, trade-off in medical screening: it's safer to over-refer than to
  miss a real case, but it does mean a certain fraction of healthy patients
  would be sent for unnecessary follow-up.
- Precision is balanced across both classes (0.91), meaning the model is
  equally reliable whichever label it predicts — it isn't just defaulting to
  "pneumonia" for everything.
- Using `class_weight` during training (to counter the imbalance between ~1,300
  NORMAL and ~3,900 PNEUMONIA training images) appears to have helped avoid the
  model collapsing to always predicting the majority class — reflected in the
  reasonably balanced precision/recall across both classes rather than one
  class being ignored.

## Limitations & Possible Improvements
- Could try transfer learning (e.g., a pretrained ResNet or VGG16 base) instead of
  training a CNN from scratch, which often performs better on small medical
  imaging datasets.
- Could experiment with different image sizes or color mode (RGB vs grayscale).
- Cross-validation would give a more robust estimate of performance than a single
  train/val/test split.
