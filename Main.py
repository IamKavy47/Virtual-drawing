import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from HandTrackingModule import HandDetector

def main():
    st.title("Virtual Painter")

    # Initialize session state variables
    if 'canvas' not in st.session_state:
        st.session_state.canvas = np.zeros((720, 1280, 3), np.uint8)
    if 'draw_color' not in st.session_state:
        st.session_state.draw_color = (255, 0, 255)
    if 'webcam_on' not in st.session_state:
        st.session_state.webcam_on = False
    if 'webcam_index' not in st.session_state:
        st.session_state.webcam_index = 0
    if 'resolution' not in st.session_state:
        st.session_state.resolution = (1280, 720)

    # Sidebar controls
    st.sidebar.header("Controls")
    brush_thickness = st.sidebar.slider("Brush Thickness", 5, 50, 25)
    eraser_thickness = st.sidebar.slider("Eraser Thickness", 50, 150, 100)
    color_choice = st.sidebar.color_picker("Choose drawing color", "#FF00FF")
    if color_choice:
        rgb = tuple(int(color_choice.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        st.session_state.draw_color = (rgb[2], rgb[1], rgb[0])

    if st.sidebar.button("Clear Canvas"):
        st.session_state.canvas = np.zeros((720, 1280, 3), np.uint8)

    # Webcam controls
    st.sidebar.subheader("Webcam Settings")
    webcam_options = [f"Webcam {i}" for i in range(5)]
    webcam_index = st.sidebar.selectbox("Select Webcam", options=webcam_options, index=st.session_state.webcam_index)
    st.session_state.webcam_index = int(webcam_index.split()[-1])

    resolutions = [(640, 480), (1280, 720), (1920, 1080)]
    resolution = st.sidebar.selectbox("Resolution", resolutions, index=resolutions.index(st.session_state.resolution))
    st.session_state.resolution = resolution

    toggle_webcam = st.sidebar.button("Toggle Webcam")
    if toggle_webcam:
        st.session_state.webcam_on = not st.session_state.webcam_on

    # Initialize hand detector
    detector = HandDetector(detectionCon=0.65)

    # Create two columns for the video feed and canvas
    col1, col2 = st.columns(2)
    video_placeholder = col1.empty()
    canvas_placeholder = col2.empty()

    # Variables for tracking previous positions
    xp, yp = 0, 0

    if st.session_state.webcam_on:
        cap = cv2.VideoCapture(st.session_state.webcam_index)
        cap.set(3, st.session_state.resolution[0])
        cap.set(4, st.session_state.resolution[1])

        while True:
            success, img = cap.read()
            if not success:
                st.error("Failed to capture frame from webcam.")
                break

            img = cv2.flip(img, 1)
            img = detector.findHands(img)
            lmList, bbox = detector.findPosition(img, draw=False)

            if lmList:
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]
                fingers = detector.fingersUp()

                if fingers[1] and fingers[2]:
                    xp, yp = 0, 0
                    cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), st.session_state.draw_color, cv2.FILLED)

                if fingers[1] and not fingers[2]:
                    cv2.circle(img, (x1, y1), 15, st.session_state.draw_color, cv2.FILLED)

                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    if st.session_state.draw_color == (0, 0, 0):
                        cv2.line(st.session_state.canvas, (xp, yp), (x1, y1), st.session_state.draw_color, eraser_thickness)
                    else:
                        cv2.line(st.session_state.canvas, (xp, yp), (x1, y1), st.session_state.draw_color, brush_thickness)

                    xp, yp = x1, y1

            imgGray = cv2.cvtColor(st.session_state.canvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, st.session_state.canvas)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            canvas_rgb = cv2.cvtColor(st.session_state.canvas, cv2.COLOR_BGR2RGB)

            video_placeholder.image(img_rgb, channels="RGB", use_container_width=True)
            canvas_placeholder.image(canvas_rgb, channels="RGB", use_container_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
