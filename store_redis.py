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

config = redis_storage.load_hash_configuration('MyHash2012_5')
lshash = None
if config is None:
    lshash = RandomBinaryProjectionTree('MyHash2012_5', 15, 30)
else:
    lshash = RandomBinaryProjectionTree(None, None, None)
    lshash.apply_config(config)

engine = Engine(DIMENSION, lshashes=[lshash], storage=redis_storage)
i = 0
for filename in os.listdir(SPATH):
    image = cv2.imread(os.path.join(SPATH, filename))
    reduced = cv2.resize(image, (32,32))
    SNAME = os.path.join(SPATH, filename)
    print(i)
    i = i + 1
    engine.store_vector(reduced.flatten(), SNAME)
redis_storage.store_hash_configuration(lshash)
