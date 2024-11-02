
---

# Image Segmentation with DenseNet121

This repository contains an end-to-end implementation of an image segmentation model using **DenseNet121** as the backbone. The model is designed to perform multi-class segmentation on images, specifically with 21 classes, utilizing the **U-Net architecture** with modifications for feature extraction, upsampling, and dense connections.

---

## Project Overview

Image segmentation is a crucial task in computer vision, allowing us to identify and label regions within an image at the pixel level. This project uses **DenseNet121** as the feature extractor in a custom U-Net-like segmentation model, leveraging pretrained weights from ImageNet and applying additional upsampling and concatenation layers to create fine-grained segmentation maps.
![image](https://github.com/user-attachments/assets/56b6f556-2567-4b1e-b88d-93a168d1f7be)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Model Architecture](#model-architecture)
- [Training Setup](#training-setup)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Getting Started

### Prerequisites

To run this project, you'll need the following:

- Python 3.x
- TensorFlow
- `segmentation_models` library
- Other dependencies, which can be installed with:

  ```bash
  pip install -r requirements.txt
  ```

### Installing Dependencies

Clone this repository, navigate to the project folder, and install dependencies:

```bash
git clone https://github.com/yourusername/segmentation-densenet121.git
cd segmentation-densenet121
pip install -r requirements.txt
```

### Dataset

To train the model, prepare or download a dataset with **21-class segmentation masks**. Organize the dataset in the following structure:

```
dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
```

Modify `train_generator` and `val_generator` paths in the code as necessary.

---

## Model Architecture

### DenseNet121 U-Net Architecture

This model leverages **DenseNet121** as the encoder (backbone) and a custom decoder for upsampling and segmentation.

- **Backbone (DenseNet121)**: Extracts features at multiple levels, pre-trained on ImageNet for improved accuracy.
- **Upsampling Decoder**: Custom upsampling layers that progressively refine feature maps.
- **Final Output**: Outputs a 256x256 segmentation map, classifying each pixel into one of the 21 classes.

#### Code for the Model

The model is defined in the `create_segmentation_model` function:

```python
def create_segmentation_model(input_shape=(256, 256, 3), num_classes=21):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the encoder
    for layer in base_model.layers:
        layer.trainable = False

    # Decoder with skip connections
    # Extract feature maps from various stages in DenseNet121
    block_3_output = base_model.get_layer("conv3_block12_concat").output
    block_4_output = base_model.get_layer("conv4_block24_concat").output
    block_5_output = base_model.get_layer("conv5_block16_concat").output

    x = layers.Conv2D(256, (3, 3), padding="same")(block_5_output)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Concatenate()([x, block_4_output])
    
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Concatenate()([x, block_3_output])
    
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)

    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model
```

---

## Training Setup

To train the model, we used:

- **Adam optimizer** for efficient parameter updates.
- **Categorical Crossentropy** as the loss function to handle multi-class classification at the pixel level.
- **MeanIoU** metric to evaluate model performance on segmentation tasks.
- **EarlyStopping** callback to prevent overfitting.

### Training Command

```python
EarlyStop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')
checkpoint_path = os.path.join(os.curdir,"checkpoint.keras")
Checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')

Tensorboard = tf.keras.callbacks.TensorBoard(board_log_path)
rl = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mean_io_u', factor=0.1, patience=5, verbose=1, mode="max", min_lr=0.0001)

MeanIou = tf.keras.metrics.MeanIoU(num_classes=21)

model.compile(optimizer='Adam'
                   ,loss='categorical_crossentropy'
                   ,metrics=[MeanIou])
history = model.fit(train_generator
                    ,validation_data=val_generator
                    ,epochs=5
                    ,callbacks=[EarlyStop,Checkpoint,Tensorboard,rl]
)

```

### Saving the Model

After training, the model can be saved as follows:

```python
model.save('model_trained.keras')
```

---

## Results

### Evaluation Metrics

The model is evaluated based on:

- **Mean Intersection over Union (MeanIoU)**: A standard metric for segmentation tasks.
- **Loss (Categorical Crossentropy)**: Measures how well the model classifies pixels.

### Visualization

You can visualize the predictions using `matplotlib` or similar libraries to inspect how well the model segments different classes.
![image](https://github.com/user-attachments/assets/6c26b7a9-552c-4de7-8e9c-6818ce11ba9d)

---

## Contributing

Feel free to open issues or pull requests if you have suggestions for improvements, bug fixes, or new features.

---

## License

This project is licensed under the MIT License. 

---

