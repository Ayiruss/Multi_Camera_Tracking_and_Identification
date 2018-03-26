from xml.dom import minidom
import cv2
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjectionTree
from redis import Redis
import os
from nearpy.storage import RedisStorage
filename = 'MVI_20011.xml'
XMLPATH='/media/student/DATA3/data_pool2/NVIDIA_AICity/XML'
IMGPATH='/media/student/DATA3/data_pool2/NVIDIA_AICity/Insight-MVT_Annotation_Train'
SPATH = 'CROPPED_2012'
DIMENSION = 3072
redis_object = Redis(host='localhost', port=6379, db=0)
redis_storage = RedisStorage(redis_object)

config = redis_storage.load_hash_configuration('MyHash2012_3')
lshash = None
if config is None:
    lshash = RandomBinaryProjections('MyHash2012_3', 10, 30)
else:
    lshash = RandomBinaryProjections(None, None)
    lshash.apply_config(config)

engine = Engine(DIMENSION, lshashes=[lshash], storage=redis_storage)


for filename in os.listdir(XMLPATH):
    doc = minidom.parse(os.path.join(XMLPATH,filename))
    FILE = filename.split('.')[0]
    image_s_path = IMGPATH + '/' + FILE + '/img'
    itemlist = doc.getElementsByTagName('frame')
    frame_id = 1
    for frame in itemlist:
        targetID = str(frame_id).zfill(5)
        image_path = image_s_path + targetID +'.jpg'
        image = cv2.imread(image_path)
        targetlist = frame.getElementsByTagName('target')
        img_id = 1
        for target in targetlist:
            box = target.getElementsByTagName('box')
            X = int(float(box[0].attributes['left'].value))
            Y = int(float(box[0].attributes['top'].value))
            W = int(float(box[0].attributes['width'].value))
            H = int(float(box[0].attributes['height'].value))
            attribute = target.getElementsByTagName('attribute')
            speed = int(float(attribute[0].attributes['speed'].value))

            if (W >= 100 and speed > 1):
                save_name = FILE + '+FRAME_' + str(frame_id) +  '_IMG_' + str(img_id) +'.jpg'
                SNAME = os.path.join(SPATH, save_name)
                print(SNAME)
                ROI = image[Y:Y+W,X:X+W]
                reducedROI = cv2.resize(ROI, (32,32))
                flatROI = reducedROI.flatten()
                cv2.imwrite(SNAME, ROI)
                engine.store_vector(flatROI, SNAME)
                cv2.rectangle(image, (X,Y), (X+W, Y+H), (0,0,255), 3)
            img_id = img_id + 1
        #print(image_path)
        cv2.imshow('image', image)
        #cv2.waitKey()
        frame_id = frame_id + 1
redis_storage.store_hash_configuration(lshash)
