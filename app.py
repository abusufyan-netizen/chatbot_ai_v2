import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os, time, json
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from drive_utils import DriveClient, drive_available
import pandas as pd

# Config
st.set_page_config(page_title='Digit Recognition AI', page_icon='🤖', layout='wide')
st.markdown("""
<style>
body, .stApp { background-color:#0e1117; color:#fafafa; }
.stButton>button{background-color:#262730;color:#fafafa;border-radius:8px;}
.stButton>button:hover{background-color:#3c4048;color:#fff;}
.card { background: #0f1720; padding: 12px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join('model','digit_recognition_model.keras')
DATA_DIR = 'data'
CORR_DIR = os.path.join(DATA_DIR, 'corrections')
MANUAL_DIR = os.path.join(DATA_DIR, 'manual_data')
HISTORY_FILE = os.path.join(DATA_DIR, 'history.json')
os.makedirs(CORR_DIR, exist_ok=True)
os.makedirs(MANUAL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

# Drive client (if configured)
drive_client = DriveClient() if drive_available() else None

# Utilities
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
    else:
        st.warning('Pretrained model not found. Place digit_recognition_model.keras in /model or use the Colab notebook.')
        return None

@st.cache_resource
def get_model():
    return load_model()

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE,'r') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE,'w') as f:
        json.dump(history, f, indent=2)
    # also upload to Drive if available
    if drive_client:
        try:
            drive_client.upload_file(HISTORY_FILE, remote_folder='history')
        except Exception:
            pass

def save_correction_image(img_pil, label):
    fname = f"corr_{label}_{int(time.time())}.png"
    path = os.path.join(CORR_DIR, fname)
    img_pil.save(path)
    # upload to Drive corrections folder too
    if drive_client:
        try:
            drive_client.upload_file(path, remote_folder='corrections')
        except Exception:
            pass
    return path

def save_manual_image(img_pil, label):
    fname = f"manual_{label}_{int(time.time())}.png"
    path = os.path.join(MANUAL_DIR, fname)
    img_pil.save(path)
    if drive_client:
        try:
            drive_client.upload_file(path, remote_folder='manual_data')
        except Exception:
            pass
    return path

# Preprocessing to match MNIST (center, invert, resize)
def preprocess_pil(pil_img):
    img = pil_img.convert('L')
    img = ImageOps.invert(img)
    img.thumbnail((20,20), Image.LANCZOS)
    new_img = Image.new('L', (28,28), color=0)
    new_img.paste(img, ((28-img.width)//2, (28-img.height)//2))
    arr = np.array(new_img).astype('float32')/255.0
    arr = arr.reshape(1,28,28,1)
    return arr, new_img

# UI Navigation
page = st.sidebar.selectbox('Select Page', ['Home','History','Performance','Manual Data'])

model = get_model()
history = load_history()

# ---------------------- HOME ----------------------
if page == 'Home':
    st.markdown('<div style="text-align:center"><h1 style="color:#00b4d8">🤖 Digit Recognition AI</h1></div>', unsafe_allow_html=True)
    st.write('Draw or upload a digit. Correct the model if it is wrong — corrections are stored and used for periodic retraining.')

    col1, col2 = st.columns([1,1])
    with col1:
        uploaded = st.file_uploader('Upload an image (png/jpg)', type=['png','jpg','jpeg'])
        st.markdown('**Or draw below:**')
        canvas = st_canvas(fill_color='rgba(255,255,255,1)', stroke_width=12, stroke_color='#ffffff', background_color='#000000', height=260, width=260, drawing_mode='freedraw', key='canvas_home')
        predict = st.button('Recognize')
    with col2:
        st.markdown('### Preview')
        preview_placeholder = st.empty()

    def pil_from_canvas(canvas_obj):
        if canvas_obj is None or canvas_obj.image_data is None:
            return None
        img = Image.fromarray(canvas_obj.image_data.astype('uint8'), 'RGBA').convert('L')
        bbox = img.getbbox()
        if bbox: img = img.crop(bbox)
        return img

    if predict:
        pil_img = None
        if uploaded:
            try:
                pil_img = Image.open(uploaded).convert('L')
            except Exception as e:
                st.error(f'Failed to open uploaded image: {e}')
        else:
            pil_img = pil_from_canvas(canvas)

        if pil_img is None:
            st.warning('No image provided. Upload or draw and then click Recognize.')
        elif model is None:
            st.error('Model not available. Place model/digit_recognition_model.keras or train and add it.')
        else:
            arr, proc = preprocess_pil(pil_img)
            preds = model.predict(arr)
            pred = int(np.argmax(preds[0])); conf = float(np.max(preds[0]))*100.0
            preview_placeholder.image(proc.resize((140,140)), caption='Processed (28x28)')

            # immediate feedback color: green if correct known (no correction yet), but status pending until user confirms
            st.success(f'Predicted: **{pred}**  — Confidence: **{conf:.2f}%**')

            # append to history as pending
            rec = { 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'predicted': int(pred), 'confidence': round(conf,2), 'correct_label': None, 'status':'Pending' }
            history.append(rec); save_history(history)

            # correction UI
            correct = st.text_input('If incorrect, enter correct digit (0-9):', key=f'corr_{len(history)}')
            if st.button('Submit Correction', key=f'sub_{len(history)}'):
                if correct.isdigit() and 0 <= int(correct) <= 9:
                    label = int(correct)
                    img_path = save_correction_image(proc, label)
                    # update last history record
                    history[-1]['correct_label'] = label
                    history[-1]['status'] = 'Correct' if label == rec['predicted'] else 'Incorrect'
                    save_history(history)
                    st.success(f'Feedback saved. Label: {label}')
                    # check if need retrain (every 5 corrections)
                    corr_files = [f for f in os.listdir(CORR_DIR) if f.endswith('.png')]
                    if len(corr_files) % 5 == 0:
                        st.info('5 corrections collected — retraining model (fine-tune). This may take a short while.')
                        from retrain import retrain_on_corrections
                        retrain_on_corrections(model_path=MODEL_PATH, corrections_folder=CORR_DIR, drive_client=drive_client)
                        # reload model
                        model = get_model()
                else:
                    st.warning('Enter a digit 0-9.')

# ---------------------- HISTORY ----------------------
if page == 'History':
    st.markdown('## 📚 Prediction History')
    if not history:
        st.info('No predictions yet.')
    else:
        df = pd.DataFrame(history)
        def row_style(r):
            if r['status'] == 'Correct': return ['background-color:#14532d;color:white']*len(r)
            if r['status'] == 'Incorrect': return ['background-color:#7f1d1d;color:white']*len(r)
            return ['background-color:#0f1720;color:#cbd5e1']*len(r)
        styled = df.style.apply(lambda row: row_style(row), axis=1)
        st.dataframe(styled, use_container_width=True)
        if st.button('Export history CSV'):
            csv = df.to_csv(index=False)
            st.download_button('Download CSV', data=csv, file_name='history.csv', mime='text/csv')

# ---------------------- PERFORMANCE ----------------------
if page == 'Performance':
    st.markdown('## 📈 Performance Monitor')
    df = pd.DataFrame(history) if history else pd.DataFrame(columns=['timestamp','predicted','confidence','correct_label','status'])
    total = len(df)
    correct_count = len(df[df['status']=='Correct'])
    incorrect_count = len(df[df['status']=='Incorrect'])
    accuracy = (correct_count/total*100) if total>0 else 0.0
    st.metric('Total Predictions', total)
    st.metric('Correct', correct_count); st.metric('Incorrect', incorrect_count)
    st.metric('Accuracy (%)', f"{accuracy:.2f}%")
    if total>0:
        chart = pd.Series(df['status'].map(lambda s: 1 if s=='Correct' else 0)).cumsum()
        st.line_chart(chart)

# ---------------------- MANUAL DATA ----------------------
if page == 'Manual Data':
    st.markdown('## 🗂️ Add Data Manually')
    up = st.file_uploader('Upload an image to add to dataset (png/jpg)', type=['png','jpg','jpeg'], key='manual_up')
    label = st.text_input('Enter label for the uploaded image (0-9)', key='manual_label')
    if st.button('Add to Manual Dataset'):
        if up and label.isdigit() and 0<=int(label)<=9:
            pil = Image.open(up).convert('L')
            path = save_manual_image(pil, int(label))
            st.success(f'Image saved to {path}')
            # if 5 manual images, retrain as well
            manual_files = [f for f in os.listdir(MANUAL_DIR) if f.endswith('.png')]
            if len(manual_files) % 5 == 0:
                st.info('5 manual images collected — retraining model (fine-tune).')
                from retrain import retrain_on_corrections
                retrain_on_corrections(model_path=MODEL_PATH, corrections_folder=MANUAL_DIR, drive_client=drive_client)
                model = get_model()
        else:
            st.warning('Provide an image and a valid label 0-9.')

# Footer
st.markdown('<hr><div style="text-align:center;color:#6b7280">Made by Abu Sufyan — Student (Organization: Abu Zar)</div>', unsafe_allow_html=True)
