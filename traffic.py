import cv2
import numpy as np
import tensorflow as tf
import time
import datetime
import requests
import sys


baseLink = 'https://tdcctv.data.one.gov.hk/{}.JPG'
# seq 1 is towards north point
# seq 2 is towards cyberport
jpgID = [
         {
            'desc': '香港島 南區 香港仔海傍道近魚市場',
            'id': 'H429F',
            'box_limit': 5, # the limit of the max size of box, larger the value dump smaller box
            'box_threshold': 50, # lowest width/height of box
            'mask_seq1': [[(0,570), (0,400), (140,300), (290,300), (250,570)]],
            'mask_seq2': [[(250,570), (290,300), (430,300), (610,470),(610,570)]]
         },
        #  {
        #     'desc': '香港島 灣仔區 香港仔隧道灣仔入口',
        #     'id': 'H210F',
        #     'box_limit': 2,
        #     'box_threshold': 2,
        #     'mask_seq1': [[(0,350), (0,300), (220,260), (390,220), (530,155), (565,195), (400,260), (250,300)]],
        #     'mask_seq2': [[(0,400), (0,330), (220,290), (390,250), (530,200), (565,245), (400,315), (250,355)]]
        #  },
         {
            'desc': '香港島 南區 香港仔隧道香港仔入口',
            'id': 'H421F',
            'box_limit': 6,
            'box_threshold': 50,
            'mask_seq1': [[(100,580), (300,250), (380,260), (320, 580)]],
            'mask_seq2': [[(0,590), (0,450), (220,250), (300,250), (100,580)]],
         },
         {
            'desc': '黃竹坑道近香港仔隧道',
            'id': 'H401F',
            'box_limit': 5,
            'box_threshold': 40,
            'mask_seq1': [[(300,590), (120,280), (210,260), (550,590)]],
            'mask_seq2': [[(550,590), (210,290), (330,250), (640,480)]],
         },
         {
            'desc': '黃竹坑道近香港仔工業學校',
            'id': 'H422F1',
            'box_limit': 8,
            'box_threshold': 30,
            'mask_seq1': [[(0,450), (0,300), (380,120), (450,200), (0,450)]],
            'mask_seq2': [[(0,450), (450,200), (500,250), (150,590)]],
         },
         {
            'desc': '英皇道近電廠街',
            'id': 'H307F',
            'box_limit': 10,
            'box_threshold': 30,
            'mask_seq1': [[(40,550), (300,250), (400,260), (300,550)]], # seq 1 not exist
            'mask_seq2': [[(40,550), (300,250), (400,260), (300,550)]],
         }
]

detector = tf.saved_model.load('faster_rcnn_inception_resnet_v2_640x640_1')

img_size = 640

def addBoundingBox(image, detection_boxes, class_ids, box_limit, box_threshold):
    rect_list = []
    for index, raw_box in enumerate(detection_boxes[0]):
        box = np.asarray(raw_box) * img_size
        id = class_ids[0][index]

        if id < 2 or id > 6:
            continue

        # remove bounding box > 1/4 screen size
        width = box[3] - box[1]
        height = box[2] - box[0]
        if height < box_threshold or width < box_threshold:
            continue
        if height > img_size / box_limit or width > img_size / box_limit:
            continue
        
        rect_list.append([int(box[1]), int(box[0]), int(width), int(height)])

    new_rect_list, _ = cv2.groupRectangles(rect_list, groupThreshold=0, eps=1)

    for new_box in new_rect_list:
        # Create a Rectangle patch
        image = cv2.rectangle(image, (new_box[0], new_box[1]), (new_box[0] + new_box[2], new_box[1] + new_box[3]), (255,0,0), 2)

    return image

def splitToTwoImage(image, index):
    # mask defaulting to black for 3-channel and transparent for 4-channel
    mask = np.zeros(image.shape, dtype=np.uint8)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]
    ignore_mask_color = (255,)*channel_count
    
    
    # Seq 1
    roi_corners = np.array(jpgID[index]['mask_seq1'], dtype=np.int32) / 640 * img_size
    roi_corners = roi_corners.astype(dtype=np.int32)
    
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    lower =(0, 0, 0) # lower bound for each channel
    upper = (0, 0, 0) # upper bound for each channel

    # create the mask and use it to change the colors
    mask_black = cv2.inRange(mask, lower, upper)
    

    # apply the mask
    image1 = cv2.bitwise_and(image, mask)
    image1[mask_black != 0] = (70,)*channel_count

    # Seq 2
    # mask defaulting to black for 3-channel and transparent for 4-channel
    mask = np.zeros(image.shape, dtype=np.uint8)

    roi_corners = np.array(jpgID[index]['mask_seq2'], dtype=np.int32) / 640 * img_size
    roi_corners = roi_corners.astype(dtype=np.int32)
    
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    # create the mask and use it to change the colors
    mask_black = cv2.inRange(mask, lower, upper)

    # apply the mask
    image2 = cv2.bitwise_and(image, mask)
    image2[mask_black != 0] = (70,)*channel_count

    return [image1, image2]

def imageInference(image, index):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector_output = detector(np.reshape(image, (1,img_size,img_size,3)))
    detection_boxes = detector_output["detection_boxes"]
    class_ids = detector_output["detection_classes"]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = addBoundingBox(
        image, 
        detection_boxes, 
        class_ids, 
        box_limit=jpgID[index]['box_limit'],
        box_threshold=jpgID[index]['box_threshold']
    )

    return image

def getContourRatio(image, name, image_logging): 
    # find contours
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([120,255,255])
    upper_blue = np.array([120,255,255])
    mask_blue = cv2.inRange(imghsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im = np.copy(image)
    cv2.drawContours(im, contours, -1, (0, 255, 0), 1)

    # calculate contours' areas
    area = 0
    for cnt in contours:
        area += cv2.contourArea(cnt)

    # find whole picture area except black
    # imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_mask = np.array([70,70,70])
    upper_mask = np.array([70,70,70])
    mask_not_black = cv2.inRange(image, lower_mask, upper_mask)
    mask_not_black = cv2.bitwise_not(mask_not_black)
    contours, _ = cv2.findContours(mask_not_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, contours, -1, (0, 255, 0), 1)
    if image_logging:
        cv2.imwrite(f'image_history/{name}.jpg', im)

    # calculate contours' areas
    total_area = 0
    for cnt in contours:
        total_area += cv2.contourArea(cnt)
    
    # print(f'traffic area: {area}, total_area: {total_area}\nratio: {area / total_area * 100}%')
    return area / total_area

def predict(index, image_logging):
    resp = requests.get(baseLink.format(jpgID[index]['id']), stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # image = cv2.GaussianBlur(image, (1,1), 0)
    image = cv2.resize(image, (img_size, img_size))

    # cv2_imshow(image) 

    image_arr = splitToTwoImage(image, index)
    raw_traffic_arr = []

    # inference and add bounding box
    for j, img in enumerate(image_arr):
        img = imageInference(img, index)
        timestring = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d;%H%M%S')
        raw_traffic_arr.append(getContourRatio(img, f'{timestring}_{index}_seq{j}', image_logging))

    return raw_traffic_arr

def get_traffic_status(image_logging):
    # print('Executing traffic retrieval')

    traffic_status = []
    for index, item in enumerate(jpgID):
        # print(item['desc'])
        result = predict(index, image_logging)
        # if not index == 5:
        #     print('Seq 1')
        #     print(f'ratio: {result[0] * 100} %')
        # print('Seq 2')
        # print(f'ratio: {result[1] * 100} %')

        # map the percentage to traffic
        # 1 is normal
        # 2 is moderate
        # 3 is severe
        state = [1, 1]
        state = list(map(lambda x: 3 if x > 0.6 else 2 if x > 0.4 else 1, result))

        traffic_status.append({
            'desc': item['desc'],
            'id': item['id'],
            'raw_percentage': result,
            'state': state
        })

    print(traffic_status)
    sys.stdout.flush()
    timestring = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    with open('./traffic.log', 'a') as f:
        f.write(f'[{timestring}]: {traffic_status}\n')
    # print('Executed traffic retrieval')

get_traffic_status(image_logging=False)
