from PIL import ImageGrab, ImageDraw
from win32gui import FindWindow, SetForegroundWindow, GetWindowRect
from time import sleep
import numpy as np
import cv2
import pytesseract
from copy import copy
import json
import itertools
import os
from pyautogui import moveTo, mouseDown, mouseUp
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

handle = FindWindow(None, 'Bluestacks App Player')

if not handle:
    exit('App not found, sry')

SetForegroundWindow(handle)
sleep(.2)
bbox = GetWindowRect(handle)
img = ImageGrab.grab(bbox)
img.save('test.png')

# locate keys
window_bounds = (bbox[2] - bbox[0], bbox[3] - bbox[1])
# key_bounds = [0.15 * window_bounds[0], 0.6 * window_bounds[1], 0.8 * window_bounds[0], window_bounds[1]]
key_bounds = [0.15 * window_bounds[0], 500, 0.8 * window_bounds[0], 900]
img_keys = img.crop(key_bounds)
img_keys_gray = np.array(img_keys.convert('L'))

draw = ImageDraw.Draw(img)

if True:
    img_keys_gray = 255 - img_keys_gray
blur = cv2.GaussianBlur(img_keys_gray, (3, 3), 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morph = cv2.morphologyEx(blur, cv2.MORPH_DILATE, kernel, iterations=1)
# thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 4)
thresh = cv2.threshold(morph, 40, 255, cv2.THRESH_BINARY)[1]

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
letters_bbox = {}

for stat, centroid in zip(list(stats), list(centroids)):
    x1 = stat[cv2.CC_STAT_LEFT]
    y1 = stat[cv2.CC_STAT_TOP]
    w = stat[cv2.CC_STAT_WIDTH]
    h = stat[cv2.CC_STAT_HEIGHT]
    
    if 10 <= w <= 90 and 40 <= h <= 90 and w < h * 1.3:
        letters_bbox[(int(centroid[0]), int(centroid[1]))] = (x1, y1, x1+w, y1+h)

        # draw.text((stat[cv2.CC_STAT_LEFT]+key_bounds[0], stat[cv2.CC_STAT_TOP]+key_bounds[1]), str(i), font=ImageFont.truetype('arial.ttf', size=30), fill='red')
        draw.rectangle((x1+key_bounds[0], y1+key_bounds[1], x1+w+key_bounds[0], y1+h+key_bounds[1]), outline='red')
        cv2.rectangle(thresh, (x1, y1), (x1+w, y1+h), color=(255,0,0))
img.save('boxes.png')
cv2.imwrite('thresh.png', thresh)

# process key values
blur = cv2.GaussianBlur(img_keys_gray, (1, 1), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

letters = set()

i = 0
for centroid, box in letters_bbox.items():
    singular_key = opening[box[1]:box[3], box[0]:box[2]]
    cv2.imwrite(f'{i}.png', singular_key)
    i += 1
    letter = pytesseract.image_to_string(singular_key, lang='eng', config='--psm 10').strip()[-1]
    
    if letter == '|':
        letters.add(('i', centroid))
    else:
        letters.add((letter.lower(), centroid))

print(letters)

class TrieNode:
    def __init__(self, prefix):
        self.prefix = prefix
        self.children = [None] * 26
        self.is_word = False
    
    def __str__(self):
        childs = ','.join([str(c) for c in self.children if c])
        if self.is_word:
            return f'*{self.prefix} + [{childs}]'
        else:
            return f'{self.prefix} + [{childs}]'
    
    def serialize(self):
        string = ''

        for i, child in enumerate(self.children):
            if child:
                if child.is_word:
                    letter = chr(i + ord('A'))
                else:
                    letter = chr(i + ord('a'))
                string += letter + child.serialize()
        
        string += ')'
        
        return string

def deserialize(is_word, prefix, it):
    node = TrieNode(prefix)
    node.is_word = is_word
    
    c = next(it)
    while c and c != ')':
        if c.isupper():
            c = c.lower()
            node.children[char(c)] = deserialize(True, prefix + c, it)
        else:
            node.children[char(c)] = deserialize(False, prefix + c, it)
        c = next(it)
    
    return node


def char(c):
    return ord(c)-ord('a')

if os.path.exists('dictionary.out'):
    with open('dictionary.out', 'r') as file:
        root = deserialize(False, '', itertools.chain.from_iterable(file))
else:
    root = TrieNode('')

    with open('wiki-100k.txt', 'r') as dictionary:
        word = dictionary.readline().strip()
        while word:
            try:
                is_valid = True
                for c in word:
                    if char(c.lower()) < 0 or char(c.lower()) >= 26:
                        is_valid = False
                        break
                if not is_valid:
                    word = dictionary.readline().strip()
                    continue
            except:
                print(word)

            current = root
            for c in word:
                if not current.children[char(c.lower())]:
                    current.children[char(c.lower())] = TrieNode(current.prefix + c.lower())
                
                current = current.children[char(c.lower())]
            
            if len(word) > 2:
                current.is_word = True
            try:
                word = dictionary.readline().strip()
            except:
                print(word)
    
    with open('dictionary.out', 'w') as file:
        file.write(root.serialize())

print('trie made')

words = set()
def bfs(current, options):
    if current.is_word:
        words.add(current.prefix)
    
    for option in options:
        if current.children[char(option)]:
            next_options = copy(options)
            next_options.remove(option)
            bfs(current.children[char(option)], next_options)

bfs(root, [tup[0] for tup in letters])


def getPos(letter, remaining):
    for l, pos in remaining:
        if letter == l:
            return pos

for word in sorted(list(words), key=lambda word: len(word)):
    print(f'doing {word}')
    remaining = letters.copy()
    for i, c in enumerate(word):
        nextPos = getPos(c, remaining)
        moveTo(bbox[0] + key_bounds[0] + nextPos[0], bbox[1] + key_bounds[1] + nextPos[1])
        remaining.remove((c, nextPos))
        if i == 0:
            mouseDown()
    
    mouseUp()
