import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
import threading
import webbrowser

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("emotion_model.h5")
labels = ['angry','disgust','fear','happy','sad','surprise','neutral']

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- SONG RECOMMENDATION ----------------
def recommend_song(emotion):
    queries = {
        "happy": "happy upbeat songs",
        "sad": "sad emotional songs",
        "angry": "calm relaxing songs",
        "fear": "peaceful instrumental music",
        "surprise": "trending songs",
        "neutral": "chill lofi music",
        "disgust": "soft instrumental music"
    }
    query = queries.get(emotion, "music")
    return f"https://www.youtube.com/results?search_query={query.replace(' ','+')}"

# ---------------- IMAGE EMOTION ----------------
def detect_emotion_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No Face Detected", ""

    x, y, w, h = faces[0]
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (48,48)) / 255.0
    roi = roi.reshape(1,48,48,1)

    pred = model.predict(roi, verbose=0)[0]
    emotion = labels[np.argmax(pred)].upper()

    link = f"""
### 🎵 Recommended YouTube Songs  
[▶ **Open {emotion} Music**]({recommend_song(emotion)})
"""

    return emotion, link

# ---------------- LIVE WEBCAM ----------------
def start_live_webcam():
    cap = cv2.VideoCapture(0)
    last_emotion = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48,48)) / 255.0
            roi = roi.reshape(1,48,48,1)

            pred = model.predict(roi, verbose=0)[0]
            emotion = labels[np.argmax(pred)]

            if emotion != last_emotion:
                webbrowser.open(recommend_song(emotion))
                last_emotion = emotion

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, emotion, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Live Emotion Detection (Press Q to Exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- THREAD ----------------
def run_webcam():
    threading.Thread(target=start_live_webcam).start()
    return "Webcam Started — Press Q to stop"

# ---------------- MODERN UI ----------------
with gr.Blocks(theme=gr.themes.Soft()) as app:

    app.css = """
    .gradio-container {max-width: 100% !important;}
    .block {padding-left: 0 !important; padding-right: 0 !important;}
    .image-container img {object-fit: cover !important;}
    """

    gr.Markdown("<h1 style='text-align:center'>🎧 AI Powered Music Generator Based on User Mood</h1>")
    gr.Markdown("<p style='text-align:center'>Upload your face image to get emotion-based music recommendations</p>")

    with gr.Tab("📷 Image Upload"):
        with gr.Row():

            with gr.Column(scale=2):
                img = gr.Image(
                    type="numpy",
                    label="Upload Face Image",
                    height=450,
                    container=False,
                    elem_classes="image-container"
                )

                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    detect_btn = gr.Button("Submit", variant="primary")

            with gr.Column(scale=1):
                emotion = gr.Textbox(label="😊 Detected Emotion", interactive=False)
                link = gr.Markdown()

        clear_btn.click(lambda: (None, "", ""), outputs=[img, emotion, link])
        detect_btn.click(detect_emotion_from_image, img, [emotion, link])

    with gr.Tab("🎥 Live Webcam"):
        gr.Markdown("""
**Before clicking:** Click the button to start live webcam  
**After clicking:** Webcam Started — Press Q to stop
""")
        status = gr.Textbox(label="Webcam Status", value="Click the button to start live webcam", interactive=False)
        webcam_btn = gr.Button("Start Live Webcam")
        webcam_btn.click(run_webcam, outputs=status)

app.launch()
