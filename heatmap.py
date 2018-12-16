import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K

def heatmap(images, model):
    
    final_images = []
    k = 1
    for i in range(len(images)):
        fig = plt.figure(figsize = (20, 6 * len(images)))
        im = np.expand_dims(images[i], axis = 0)
        preds = model.predict(im)
        output = model.get_layer("dense_2").output[:, np.argmax(preds)]
        last_conv_layer = model.get_layer('conv2d_2')
        grads = K.gradients(output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis = (0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        grads_value, conv_layer_output_value = iterate([im])
        for j in range(conv_layer_output_value.shape[2]):
            conv_layer_output_value[:, :, j] *= grads_value[j]
        heatmap_1 = np.mean(conv_layer_output_value, axis =- 1)
        heatmap_1 = np.maximum(heatmap_1, 0)
        heatmap_1 /= np.max(heatmap_1)
        fig.add_subplot(len(images), 3, k)
        plt.imshow(heatmap_1)
        
        heatmap_2 = cv2.resize(heatmap_1, (images[i].shape[1], images[i].shape[0]))
        heatmap_2 = np.uint8(255*heatmap_2)
        heatmap_2 = cv2.applyColorMap(heatmap_2, cv2.COLORMAP_JET)
        fig.add_subplot(len(images), 3, k + 1)
        plt.imshow(heatmap_2)
        
        superimposed_img = heatmap_2 * 0.7 + images[i] * 255
        superimposed_img = np.array(superimposed_img).astype('uint8')
        fig.add_subplot(len(images), 3, k + 2)
        plt.imshow(superimposed_img)
        final_images.append(superimposed_img)
        k += 3
    
    return np.array(final_images)
    