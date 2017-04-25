import urllib.request
import cv2
import os
from py_ms_cognitive import PyMsCognitiveImageSearch
from glob import glob
from config import api_key, cascade

# GLOBAL
font = cv2.FONT_HERSHEY_SIMPLEX
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

#path = 'swift/'

'''
dict_keys(['thumbnail_url','host_page_url', 'content_url','content_size',
'image_insights_token', 'image_id', 'json', 'name', 'web_search_url'])
'''


def download_images():
    if not os.path.exists('swift', ):
        print("creating swift folder")
        os.makedirs('swift')

    pic_num = len(os.listdir('swift/')) + 1
    search_term = 'Taylor Swift'
    search_service = PyMsCognitiveImageSearch(api_key, search_term, silent_fail=True)
    first = search_service.search(limit=50, format='json')  # 1-50
    second = search_service.search(limit=50, format='json')  # 51-100
    third = search_service.search(limit=50, format='json')  # 51-100
    fourth = search_service.search(limit=50, format='json')  # 51-100
    fifth = search_service.search(limit=50, format='json')  # 51-100

    for i in first:
        save_loc = 'swift/{}.{}'.format(str(pic_num), i.json['encodingFormat'])
        # gray_loc = 'gray/{}.{}'.format(str(pic_num), i.json['encodingFormat'])
        try:
            urllib.request.urlretrieve(i.content_url, save_loc)
            pic_num += 1
            print('found an image {}'.format(pic_num))
        except Exception as e:
            print("we have an error: {}".format(e))

    for i in second:
        save_loc = 'swift/{}.{}'.format(str(pic_num), i.json['encodingFormat'])
        # gray_loc = 'gray/{}.{}'.format(str(pic_num), i.json['encodingFormat'])
        try:
            urllib.request.urlretrieve(i.content_url, save_loc)
            pic_num += 1
            print('found an image {}'.format(pic_num))
        except Exception as e:
            print("we have an error: {}".format(e))

    for i in third:
        save_loc = 'swift/{}.{}'.format(str(pic_num), i.json['encodingFormat'])
        # gray_loc = 'gray/{}.{}'.format(str(pic_num), i.json['encodingFormat'])
        try:
            urllib.request.urlretrieve(i.content_url, save_loc)
            pic_num += 1
            print('found an image {}'.format(pic_num))
        except Exception as e:
            print("we have an error: {}".format(e))

    for i in fourth:
        save_loc = 'swift/{}.{}'.format(str(pic_num), i.json['encodingFormat'])
        # gray_loc = 'gray/{}.{}'.format(str(pic_num), i.json['encodingFormat'])
        try:
            urllib.request.urlretrieve(i.content_url, save_loc)
            pic_num += 1
            print('found an image {}'.format(pic_num))
        except Exception as e:
            print("we have an error: {}".format(e))

    for i in fifth:
        save_loc = 'swift/{}.{}'.format(str(pic_num), i.json['encodingFormat'])
        # gray_loc = 'gray/{}.{}'.format(str(pic_num), i.json['encodingFormat'])
        try:
            urllib.request.urlretrieve(i.content_url, save_loc)
            pic_num += 1
            print('found an image {}'.format(pic_num))
        except Exception as e:
            print("we have an error: {}".format(e))

            # return os.listdir('swift/')
def get_face(path):
    '''
    iterates over images, finds and copies faces to face/ dir.
    :param path: location of images
    :return:
    '''
    face_cascade = cv2.CascadeClassifier(cascade)  # CASCADE CLASSIFIER

    if not os.path.exists('face'):
        print("creating face folder")
        os.makedirs('face')
    else:
        print('face folder is here')

    num = len(os.listdir('face/')) + 1
    loc = 'face/'
    for fn in glob(path + '/*'):
        img = cv2.imread(fn)
        gray = cv2.imread(fn, 0)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,
                                              scaleFactor=1.1,
                                              minNeighbors=6,
                                              minSize=(30, 30))
        for (x, y, w, h) in faces:
            print("found a face:{}".format(num))
            # cv2.rectangle(img, (x, y), (x + w, y + h), RED, 2)  # draw rectangle
            roi_color = img[y:y + h, x:x + w]
            # resize_image = cv2.resize(img, (100,100))
            cv2.imwrite(loc + str(num) + '.png', roi_color)
        num += 1
    print('done finding faces')


# show = get_face(path)
# print(show)
# download_images()


def request_images(times, search_term):
    '''
    :param times: 50x
    :param search_term: what images you want to search
    :return: search_term dir
    '''
    save_path = search_term.replace(" ", "_")
    if not os.path.exists(save_path):
        print("creating: {} dir.".format(save_path))
        os.makedirs(save_path)
    else:
        print('{} : in directory'.format(save_path))

    pic_num = len(os.listdir(save_path)) + 1
    # search_term = 'Taylor Swift'
    search_service = PyMsCognitiveImageSearch(api_key, search_term, silent_fail=True)
    for event in range(times):
        event = search_service.search(limit=50, format='json')  # 1-50
        for i in event:
            save_loc = save_path + '/{}.{}'.format(str(pic_num), i.json['encodingFormat'])
            try:
                urllib.request.urlretrieve(i.content_url, save_loc)
                pic_num += 1
                print('found an image {}'.format(pic_num))
            except Exception as e:
                print("we have an error: {}".format(e))
    return save_path


def get_face(path):
    '''
    iterates over images, finds and copies faces to face/ dir.
    :param path: location of images
    :return:
    '''

    save_path = path + '_face/'
    face_cascade = cv2.CascadeClassifier(cascade)  # CASCADE CLASSIFIER

    if not os.path.exists(save_path):
        print("creating face folder")
        os.makedirs(save_path)
    else:
        print('face folder is here')

    num = len(os.listdir(save_path)) + 1
    # loc = 'face/'
    for fn in glob(path + '/*'):
        img = cv2.imread(fn)
        gray = cv2.imread(fn, 0)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,
                                              scaleFactor=1.1,
                                              minNeighbors=6,
                                              minSize=(30, 30))
        for (x, y, w, h) in faces:
            print("found a face:{}".format(num))
            # cv2.rectangle(img, (x, y), (x + w, y + h), RED, 2)  # draw rectangle
            roi_color = img[y:y + h, x:x + w]
            # resize_image = cv2.resize(img, (100,100))
            cv2.imwrite(save_path + str(num) + '.png', roi_color)
        num += 1
    print('done finding faces')

find = request_images(2, 'Taylor Swift')

get_face(find)
