import av
import os 
import cv2
import streamlit as st
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model
from collections import deque
import imageio 


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # perform edge detection
    #img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


def app():
    def in_recorder_factory() -> MediaRecorder:
        return MediaRecorder(
            "input.mp4", format="mp4"
        )

    def out_recorder_factory() -> MediaRecorder:
        return MediaRecorder("output.flv", format="flv")
    
    st.title("Lip Reading")
    
    webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": True,
            "audio": True,
        },
        video_frame_callback=video_frame_callback,
        in_recorder_factory=in_recorder_factory,
        #out_recorder_factory=out_recorder_factory,
    )

    st.info('This is all the machine learning model sees when making a prediction')


    video, annotations = load_data(tf.convert_to_tensor("input.mp4"))
    imageio.mimsave('animation.gif', video, fps=10)
    st.image('animation.gif', width=400) 

    st.info('This is the output of the machine learning model as tokens')
    model = load_model()
    yhat = model.predict(tf.expand_dims(video, axis=0))
    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    st.text(decoder)

    #Convert prediction to text
    st.info('Decode the raw tokens into words')
    converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
    st.text(converted_prediction)


if __name__ == "__main__":
    app()