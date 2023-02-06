import subprocess
import cv2
rtmp_url = "rtmp://203.91.121.211:1935/livestream/"

cap = cv2.VideoCapture(rtmp_url)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


print(fps, width, height)