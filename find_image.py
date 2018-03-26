from xml.dom import minidom
import cv2
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from redis import Redis
import os
from nearpy.distances import ManhattanDistance, CosineDistance, EuclideanDistance
from nearpy.storage import RedisStorage
from nearpy.filters import NearestFilter
from skimage.measure import structural_similarity as ssim
# Dimension of our vector space
dimension = 3072
redis_object = Redis(host='localhost', port=6379, db=0)
redis_storage = RedisStorage(redis_object)

config = redis_storage.load_hash_configuration('MyHash2012_5')
lshash = None
lshash = RandomBinaryProjections(None, None)
lshash.apply_config(config)
sift = cv2.xfeatures2d.SIFT_create()
#rbp = RandomBinaryProjections('MyHash2012', 10)
engine = Engine(dimension, lshashes=[lshash], storage=redis_storage, distance=CosineDistance(), vector_filters=[NearestFilter(30)])
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
bf_ham = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
orb = cv2.ORB()
image = cv2.imread('CROPPED/MVI_20011+FRAME_183_IMG_1.jpg')
resized = cv2.resize(image, (32,32))
kp1, desc1 = sift.detectAndCompute(image, None)
#okp1, odesc1 = orb.detectAndCompute(image, None)
# Create random query vector

# Get nearest neighbours
MAX = 0.0
cv2.imshow('SEARCH', image)
cv2.waitKey()
match_image = None
N = engine.neighbours(resized.flatten())
for neighbors in N:
    result = cv2.imread(neighbors[1])
    result_resized = cv2.resize(result, (32,32))
    kp2, desc2 = sift.detectAndCompute(result,None)
    MIN = min(len(desc1), len(desc2))
    print(MIN)
    #okp2, odesc2 = orb.detectAndCompute(result, None)
    matches = bf.match(desc1,desc2)
    #matches = sorted(matches, key = lambda x:x.distance)
    total = len(matches)
    percent = float(float(total)/float(len(desc1)))
    print('SIFT - ', total, 'Percentage : - ', percent)
    #matches = bf_ham.match(odesc1, odesc2)
    if percent > MAX:
        MAX = percent
        match_image = result
    #print('ORB - ', len(matches))
    #s = ssim(resized, result_resized, multichannel=True)
    res = cv2.matchTemplate(resized,result_resized,cv2.TM_CCORR_NORMED)
    #print(res)
    cv2.imshow('result', result)
    print(neighbors[1])
    cv2.waitKey()
cv2.imshow('match', match_image)
cv2.waitKey()
cv2.destroyAllWindows()
#print(N)
