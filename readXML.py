from xml.dom import minidom
import cv2
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from redis import Redis
import os
from nearpy.storage import RedisStorage
filename = 'MVI_20011.xml'
XMLPATH='/home/student/data_pool2/NVIDIA_AICity_Dataset/DETRAC-Train-Annotations-XML'
IMGPATH='/home/student/data_pool2/NVIDIA_AICity_Dataset/Insight-MVT_Annotation_Train'
SPATH = 'CROPPED'
DIMENSION = 3072
redis_object = Redis(host='localhost', port=6379, db=0)
redis_storage = RedisStorage(redis_object)

config = redis_storage.load_hash_configuration('MyHash')
lshash = None
if config is None:
    lshash = RandomBinaryProjections('MyHash', 10)
else:
    lshash = RandomBinaryProjections(None, None)
    lshash.apply_config(config)

engine = Engine(DIMENSION, lshashes=[lshash], storage=redis_storage)


for filename in os.listdir(XMLPATH):
    doc = minidom.parse(os.path.join(XMLPATH,filename))
    FILE = filename.split('.')[0]
    image_s_path = IMGPATH + '/' + FILE + '/img'
    itemlist = doc.getElementsByTagName('frame')
    #print(len(itemlist))
    frame_id = 1
    for item in itemlist:
        targetlist = item.getElementsByTagName('target')
        #target_id = target.getElementsByTagName('target')
        img_id = 1
        for target in targetlist:
            targetID = str(img_id).zfill(5)
            image_path = image_s_path + targetID +'.jpg'
            image = cv2.imread(image_path)
            box = target.getElementsByTagName('box')
            X = int(float(box[0].attributes['left'].value))
            Y = int(float(box[0].attributes['top'].value))
            W = int(float(box[0].attributes['width'].value))
            H = int(float(box[0].attributes['height'].value))
            attribute = target.getElementsByTagName('attribute')
            speed = int(float(attribute[0].attributes['speed'].value))

            if (W >= 100 and speed > 1):

                ROI = image[Y:Y+W,X:X+W]
                cv2.imshow('ROI', ROI)
                save_name = FILE + '+FRAME_' + str(frame_id) +  '_IMG_' + str(img_id) +'.jpg'
                SNAME = os.path.join(SPATH, save_name)
                #cv2.imwrite(SNAME, ROI)
                cv2.rectangle(image, (X,Y), (X+W, Y+H), (0,0,255), 3)
                reducedROI = cv2.resize(ROI, (32,32))
                flatROI = reducedROI.flatten()
                #print('IMAGE_S_PATH - ' , image_s_path)
                #print('image_path - ', image_path)
                #print('store_path - ', SNAME)
                #engine.store_vector(flatROI, SNAME)
            print(image_path)
            cv2.imshow('image', image)
            cv2.waitKey()
            img_id = img_id + 1
        frame_id = frame_id + 1


#redis_storage.store_hash_configuration(lshash)
