import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT
from boxmot import StrongSORT
from ultralytics import YOLO

blue = [255,0,0]
green = [0,255,0]
red = [0,0,255]
font_thickness=3
font_scale = 1


tracker = StrongSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    device='cpu',
    fp16=False,
)

model = YOLO('yolov8n.pt') # pass any model type

vid = cv2.VideoCapture(0)

while True:
    ret, im = vid.read()
    dets = []
    
    results = model.predict(source=im, save=False, stream=False, verbose=False, show=True)
    for i,box in enumerate(results[0].boxes):
        cls = results[0].names[int(box.cls)]
        cls = cls.replace("class",'')
        if cls.isnumeric():
            cls = int(cls)
            
        x, y, w, h = box.xywh.int().tolist()[0]
        # print(f'{im.shape=} {cls=}, {box.xyxy=}, {box.xyxyn=}, {box.xywh=}, {box.xywhn=}, {box.data=}')
        # id = results[0].boxes.id
        if cls=='person':
            bbox = box.xyxy.squeeze().to(int).tolist()
            # print(f'{bbox=}')
            dets.append([*bbox, float(box.conf), 0])
    # substitute by your object detector, output has to be N X (x, y, x, y, conf, cls)
    # print(dets)
    dets = np.array(dets)
    if len(dets) == 0:
        continue
    tracks = tracker.update(dets, im) # --> M X (x, y, x, y, id, conf, cls, ind)
    # print(f'{tracks=}')
    for track in tracks:
        xyxy = track[:4].astype('int')
        id = track[4].astype('int')
        print(f'{xyxy=}, {id=}')
        color = red
        text_x = xyxy[0]
        text_y = xyxy[1 ]
        text = f'The id is = {id}'
        cv2.rectangle(im, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness=font_thickness)
        cv2.putText(im, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    # tracker.plot_results(im, show_trajectories=True)
    # print(tracks)

    # break on pressing q or space
    cv2.imshow('BoxMOT detection', im)     
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()