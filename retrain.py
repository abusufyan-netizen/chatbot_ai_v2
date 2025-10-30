import os, glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images_from_folder(folder):
    X, y = [], []
    for f in os.listdir(folder):
        if f.endswith('.png'):
            try:
                # filename format: corr_label_timestamp.png or manual_label_timestamp.png
                parts = f.split('_')
                label = int(parts[1]) if len(parts)>1 and parts[1].isdigit() else 0
            except:
                label = 0
            img = Image.open(os.path.join(folder,f)).convert('L').resize((28,28))
            arr = np.array(img).astype('float32')/255.0
            X.append(arr); y.append(label)
    if len(X)>0:
        X = np.array(X).reshape(-1,28,28,1)
        y = np.array(y)
    else:
        X = np.empty((0,28,28,1)); y = np.empty((0,))
    return X, y

def retrain_on_corrections(model_path, corrections_folder, drive_client=None):
    if not os.path.exists(model_path):
        print('Base model not found:', model_path); return
    model = tf.keras.models.load_model(model_path)
    X, y = load_images_from_folder(corrections_folder)
    if len(X)==0:
        print('No correction images found.'); return
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
    datagen.fit(X)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(datagen.flow(X,y, batch_size=16), epochs=3, verbose=2)
    model.save(model_path)
    if drive_client:
        try:
            drive_client.upload_file(model_path, remote_folder='models')
        except Exception:
            pass
    print('Retraining complete and model saved.')
