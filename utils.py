import cv2
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def generate_gradcam(model, img_path, layer_name="conv5_block3_out"):

    # Load the same base ResNet50 (will share weights for Grad-CAM only)
    base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    # Get the target layer
    target_layer = base_model.get_layer(layer_name)

    # Create a model that maps input -> target_layer + output
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[target_layer.output, base_model.output]
    )

    # Preprocess image
    img_array = preprocess_image(img_path)

    # Gradient computation
    with tf.GradientTape() as tape:
        conv_outputs, _ = grad_model(img_array)
        pooled_grads = tf.reduce_mean(tape.gradient(_, conv_outputs), axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    # Load original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    # Apply heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    gradcam_path = "gradcam_output.jpg"
    cv2.imwrite(gradcam_path, superimposed_img)
    return gradcam_path

