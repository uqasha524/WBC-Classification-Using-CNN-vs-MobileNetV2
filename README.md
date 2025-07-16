# 🧬 WBC Classification Using CNN vs Transfer Leraning

## 📌 Project Overview

This project aims to classify human white blood cell (WBC) types using two different deep learning approaches:

A custom-built Convolutional Neural Network (CNN) model

Multiple Transfer Learning models based on DenseNet, MobileNetV2, VGG16, and Vision Transformer (ViT_B16)

The objective is to compare the performance, generalization capability, and efficiency of traditional CNNs with advanced pre-trained architectures on a real-world microscopic blood smear image dataset.

---

## 🩸 What Are White Blood Cells?

White blood cells (WBCs) are essential components of the human immune system. There are 5 primary types:
- **Neutrophil** 🟠
- **Eosinophil** 🟠
- **Basophil** 🔵
- **Monocyte** 🟤
- **Lymphocyte** 🟣

Accurate classification helps pathologists detect infections, allergies, leukemia, and other disorders.

---
<img src="Image/main.png" alt="Stone Paper Scissors Game Demo" width="600"/>

---

## 🗂️ Dataset Structure

The dataset contains labeled microscopic images of white blood cells, organized like this:

```plaintext
Dataset/
├── Train/
│   ├── Neutrophil/
│   ├── Eosinophil/
│   ├── Basophil/
│   ├── Monocyte/
│   └── Lymphocyte/
├── Test/
│   └── ...
├── Balanced_Train/   # ✅ Automatically generated (see below)

⚠️ Balanced_Train is excluded from this repo due to size.
You can generate it using the code or request download access.
```
---

## Preprocessing & Balancing Strategy
The original training data in Dataset/Train/ was imbalanced, with some WBC types like Basophils having very few images compared to Neutrophils.
To solve this, we created a new folder Balanced_Train/ by:

1. Copying original images of each class
2. Augmenting underrepresented classes using:
3. Rotation (±15°)
4. Zoom (±10%)
5. Horizontal flip
6. Nearest-pixel filling

All classes were augmented up to the size of the majority class (Neutrophil: 6231 images).
``` The Balanced_Train/ folder is ignored from GitHub, but you can regenerate it using the notebook with just one click.```

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import os, shutil

source_dir = "../Dataset/Train"
balanced_dir = "../Dataset/Balanced_Train"

# Create balanced dir
os.makedirs(balanced_dir, exist_ok=True)

# Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Count class images
class_counts = {}
for class_name in os.listdir(source_dir):
    path = os.path.join(source_dir, class_name)
    class_counts[class_name] = len(os.listdir(path))

target_count = max(class_counts.values())

# Balance classes
for class_name, count in class_counts.items():
    src = os.path.join(source_dir, class_name)
    dst = os.path.join(balanced_dir, class_name)
    os.makedirs(dst, exist_ok=True)

    # Copy originals
    for fname in os.listdir(src):
        shutil.copy(os.path.join(src, fname), os.path.join(dst, fname))

    # Generate augmented images
    if count < target_count:
        needed = target_count - count
        generated = 0
        image_files = os.listdir(src)

        while generated < needed:
            for fname in image_files:
                if generated >= needed:
                    break

                img = load_img(os.path.join(src, fname))
                x = img_to_array(img).reshape((1,) + img.size + (3,))

                for batch in datagen.flow(x, batch_size=1):
                    array_to_img(batch[0]).save(os.path.join(dst, f"aug_{generated}_{fname}"))
                    generated += 1
                    break
```
Project Structure:
```plaintext
.
├── Dataset/          # WBC image folders (Train, Test, Balanced_Train)
├── Model/            
│   └── cnn_1.h5
│   └── cnn_2.h5
│   └── cnn_3.h5
│   └── dense_freeze.h5
│   └── dense_freeze_1.h5
│   └── mobile_1.h5
│   └── vgg16_freeze.keras
│   └── ViT_B16.keras
├── Notebook/
│   └── CNN-Convolutional_Neural_Network.ipynb
│   └── Dense_Net_Transfer_Learning.ipynb
│   └── Mobile_Net_Transfer_Learning.ipynb
│   └── VGG-16-transfer-learning.ipynb
│   └── ViT_B16.ipynb
├── Image/
│   └── main.png     
└── README.md         
```
---
## 🧠 Model Approaches

To classify white blood cell (WBC) types, we explored two deep learning strategies:

Custom CNN – A convolutional neural network built from scratch, tailored and optimized specifically for our WBC image dataset.

Transfer Learning – Leveraged multiple state-of-the-art pre-trained models including MobileNetV2, DenseNet, VGG16, and ViT_B16. These models, originally trained on the ImageNet dataset, were fine-tuned or used as feature extractors to improve classification performance.

All models were trained and evaluated on a balanced dataset and compared based on their classification accuracy, efficiency, and ability to generalize to unseen data.

---

### ✅ 1. Custom CNN Architecture

The custom CNN was designed specifically for image-based classification of white blood cells with the following key components:

- **Input Rescaling**: Pixel values normalized using `Rescaling(1./255)`
- **5 Convolutional Layers** with:
  - Increasing filter sizes: `4 → 8 → 16 → 32 → 64`
  - Stride of `2` for downsampling
  - `ReLU` activation
  - `padding='same'` to preserve feature map size
- **Batch Normalization** after each convolutional and dense layer for faster convergence
- **Flatten layer** before classification head
- **Dense Layers**:
  - One hidden dense layer with 8 units (`ReLU`)
  - Final output layer with 5 units (`softmax`) for classifying:  
    **Neutrophil, Eosinophil, Basophil, Monocyte, Lymphocyte**

> ✅ Achieved **93% test accuracy** after training on the balanced dataset

### ⚠️ Transfer Learning: Not Always Superior

While transfer learning using pre-trained models such as MobileNetV2, DenseNet, VGG16, and ViT_B16 can be powerful, it's essential to consider their domain applicability:

These models are trained on ImageNet, a dataset containing general-purpose object categories—not specialized domains like microscopic white blood cell images.

As a result, when the target dataset has domain-specific features, pre-trained representations may require extensive fine-tuning or may not generalize optimally.

To explore this, we compared a custom CNN model with several transfer learning architectures. The results are summarized below:

| Model       | Test Accuracy | Generalization | Training Time |
| ----------- | ------------- | -------------- | ------------- |
| Custom CNN  | ✅ 93%        | High           | Moderate      |
| MobileNetV2 | ✅ 89%        | Medium         | Faster        |
| DenseNet    | ✅ 84.44%     | Low            | Moderate      |
| VGG16       | ✅ 94.65%     | High           | Slow          |
| Vit B16     | ✅ **96.20%** | High           | Slower        |

🔍 Despite initial expectations, our custom CNN outperformed some transfer learning models in terms of generalization and accuracy. However, ViT_B16 and VGG16 surpassed all models in test accuracy, albeit at the cost of longer training times.

This highlights the importance of experimenting with both custom-built models and various transfer learning architectures, especially in medical imaging, where domain-specific patterns often require tailored solutions.


---

## 📦 Requirements

> ⚠️ **Important Note:**  
> TensorFlow is **not yet compatible with Python 3.12+**.  
> Please use **Python 3.9 to 3.11** for smooth execution.

### ✅ Supported Python Version

```bash
python==3.10  # Or any version between 3.9 and 3.11

Python Libraries Used
The following libraries were used to build, train, and evaluate the models:

TensorFlow & Keras – For deep learning and model training

NumPy – For numerical operations

Matplotlib – Plotting accuracy/loss curves

Seaborn – Visualization of confusion matrix

scikit-learn – Evaluation metrics like classification report, confusion matrix

Pillow (PIL) – Image handling and preprocessing

shutil, os, collections – File/directory manipulation

```
## **Contact**
Made with ❤️ by **Uqasha Zahid**
📫 uqashazahid@gmail.com
