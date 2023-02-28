import cv2
import time
import HandTrackingModule as htm
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


capt = cv2.VideoCapture(0)
capt.set(3,640)
capt.set(4,480)

detector = htm.handDetector(detectionCon=0.6)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol = volume.GetVolumeRange()
max_vol = vol[1]
min_vol = vol[0]

ptime = 0

while True:
    success,img = capt.read()
    # find hand & positions of 21 landmark and
    # store those in lm_positions_list
    img = detector.findHands(img)
    lm_positions_list, bbox = detector.findPosition(img, draw=True)
    if len(lm_positions_list) != 0:
        # filter based on size
        area = ((bbox[2] - bbox[0])*(bbox[3] - bbox[1]))//100
        if 300 < area < 1000:
            # finding distance between thumb and index finger
            length, img, lineinfo = detector.findDistance(4, 8, img)

            # getting interpolation points
            volm = np.interp(length, (15, 200), (min_vol, max_vol))
            volm_bar = np.interp(length, (15, 200), (400, 150))
            volm_per = np.interp(length, (15, 200), (0, 100))
            volume.SetMasterVolumeLevel(volm, None)

            # smoothening
            smoothness = 8
            vol_per = smoothness * round(volm_per / smoothness)

            # checking which finger is up
            fingers = detector.fingersUp()

            # if pinky is down set the volume
            if fingers[4] == 1:
                volume.SetMasterVolumeLevelScalar(vol_per / 100, None)

                cv2.circle(img, (lineinfo[4],lineinfo[5]), 10, (0,255,0), cv2.FILLED)
            else:
                cv2.circle(img, (lineinfo[4], lineinfo[5]), 10, (255,0,0), cv2.FILLED)


            # designing ractangular vol bar
            cv2.rectangle(img, (50, int(volm_bar)), (80, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f"{vol_per}%", (40, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)

    cv2.rectangle(img, (50,150), (80,400), (255,0,0), 3)
    gVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Vol Set: {gVol}', (400, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255,0,0), 2)


    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, f"FPS: {int(fps)}", (40, 30), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 2)

    cv2.imshow("webcam",img)
    cv2.waitKey(1)
