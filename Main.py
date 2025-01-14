import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
from HandTrackingModule import HandDetector


class VirtualPainterTransformer(VideoTransformerBase):
    def __init__(self):
        self.canvas = np.zeros((720, 1280, 3), np.uint8)
        self.detector = HandDetector(detectionCon=0.65)
        self.draw_color = (255, 0, 255)
        self.brush_thickness = 25
        self.eraser_thickness = 100
        self.xp, self.yp = 0, 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = self.detector.findHands(img)
        lmList, bbox = self.detector.findPosition(img, draw=False)

        if lmList:
            x1, y1 = lmList[8][1:]  # Index finger tip
            x2, y2 = lmList[12][1:]  # Middle finger tip
            fingers = self.detector.fingersUp()

            # Selection mode: both index and middle fingers are up
            if fingers[1] and fingers[2]:
                self.xp, self.yp = 0, 0
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), self.draw_color, cv2.FILLED)

            # Drawing mode: only index finger is up
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, self.draw_color, cv2.FILLED)

                if self.xp == 0 and self.yp == 0:
                    self.xp, self.yp = x1, y1

                if self.draw_color == (0, 0, 0):  # Eraser
                    cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), self.draw_color, self.eraser_thickness)
                else:
                    cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), self.draw_color, self.brush_thickness)

                self.xp, self.yp = x1, y1

        # Combine canvas with the webcam feed
        imgGray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, self.canvas)

        return img


def main():
    st.title("Virtual Painter with WebRTC")

    # Sidebar controls
    st.sidebar.header("Controls")
    color_choice = st.sidebar.color_picker("Choose drawing color", "#FF00FF")
    rgb = tuple(int(color_choice.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    draw_color = (rgb[2], rgb[1], rgb[0])

    brush_thickness = st.sidebar.slider("Brush Thickness", 5, 50, 25)
    eraser_thickness = st.sidebar.slider("Eraser Thickness", 50, 150, 100)

    # WebRTC streamer
    ctx = webrtc_streamer(
        key="virtual-painter",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=VirtualPainterTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

    # Update drawing settings
    if ctx.video_transformer:
        ctx.video_transformer.draw_color = draw_color
        ctx.video_transformer.brush_thickness = brush_thickness
        ctx.video_transformer.eraser_thickness = eraser_thickness


if __name__ == "__main__":
    main()
