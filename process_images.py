from keras.applications import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import h5py
from scipy.misc import imread, imresize
import numpy as np
import os

#Building the model
input_tensor = Input(shape=(116,116,3))
base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

# build the classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.4))
top_model.add(Dense(1024, activation = 'relu'))
top_model.add(Dense(1, activation='sigmoid'))

model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
#Adding the weights to the model
model.load_weights('./vgg16_weights_best.h5')


paths = ["images_resize/" + path for path in sorted(os.listdir("images_resize/"))]
batch_size = 16
out_tensors = np.zeros((len(paths), 2048), dtype="float32")
print(out_tensors.shape)
for idx in range(len(paths) // batch_size + 1):
    batch_bgn = idx * batch_size
    batch_end = min((idx+1) * batch_size, len(paths))
    imgs = []
    for path in paths[batch_bgn:batch_end]:
        img = imread(path)
        img = imresize(img, (116,116)).astype("float32")
        img = preprocess_input(img[np.newaxis])
        imgs.append(img)
    batch_tensor = np.vstack(imgs)
    print("tensor", idx, "with shape",batch_tensor.shape)
    out_tensor = base_model.predict(batch_tensor, batch_size=32)
    print("output shape:", out_tensor.shape)
    out_tensors[batch_bgn:batch_end, :] = out_tensor
print("shape of representation", out_tensors.shape)

# Serialize representations
h5f = h5py.File('img_emb.h5', 'w')
h5f.create_dataset('img_emb', data=out_tensors)
h5f.close()
