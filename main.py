

import pylab as plt
import matplotlib.pyplot as plt
import math
import os
import matplotlib.ticker as plticker
try:
    from PIL import Image
except ImportError:
    import Image
import cv2
import numpy as np


class ImageProcessor:

    def drawStripes(filename, dx=100, dy=100, grid_color=[0,0,0]):
        img = plt.imread(filename)
        img[:,::dy,:] = grid_color
        #img[::dx,:,:] = grid_color
        plt.imshow(img)
        plt.show()
    #drawStripes('5angles.png')

    """
    def rotateImage(filename, angle):
        image = cv2.imread(filename)
        originalH, originalW, channels = image.shape
    
    
    
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        #angle = 30
        scale = 1
    
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        #rotated = cv2.resize(rotated, (originalW, originalH))
        print(rotated.shape)
    
        cv2.imshow('original Image', image)
        cv2.imshow('Rotated Image', rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    #rotateImage('5angles.png', 90)
    
    """
    def  rotate_image(filename, angle):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """
        mat = cv2.imread(filename)
        height, width = mat.shape[:2] # image shape has 3 dimensions
        image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0])
        abs_sin = abs(rotation_mat[0,1])
        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]
        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),borderValue=(255, 255, 255))

        w1 = int(abs(width*math.cos(math.radians(angle)))+abs(height*math.sin(math.radians(angle))))
        h1 = int(abs(width*math.sin(math.radians(angle)))+abs(height*math.cos(math.radians(angle))))

        rotated_mat = cv2.resize(rotated_mat, (w1, h1))
        #print(width, height)
        #print(w1, h1)
        #print(math.cos(math.radians(angle)))
        #cv2.imshow('original Image', mat)
        #cv2.namedWindow('Rotated Image, angle='+str(angle), cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Rotated Image', width, height)


        cv2.imshow('Rotated Image, angle='+str(angle), rotated_mat)
        cv2.imwrite(filename, rotated_mat)

        #cv2.imwrite('rotated.png', rotated_mat)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #return rotated_mat

    #rotate_image('5angles.png', 180)
    """
    from PIL import Image
    filename = '5angles.png'
    img = Image.open(filename)
    img.load()
    print(type(img))
    img.show()"""


    """def extract(filename):
        im = cv2.imread(filename)
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv.drawContours(img, contours, -1, (0, 255, 0), 3)
        cnt = contours[4]
        cv2.drawContours(im, [cnt], 0, (0, 255, 0), 3)
        cv2.imshow('fd', im)
        cv2.waitKey(0)
    extract('2donuts.PNG')"""

    """def getContours(filename):
        hsv_min = np.array((2, 28, 65), np.uint8)
        hsv_max = np.array((26, 238, 255), np.uint8)
        fn = filename  # путь к файлу с картинкой
        img = cv2.imread(fn)
    
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # меняем цветовую модель с BGR на HSV
        thresh = cv2.inRange(hsv, hsv_min, hsv_max)  # применяем цветовой фильтр
        # ищем контуры и складируем их в переменную contours
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        # отображаем контуры поверх изображения
        cv2.drawContours(img, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)
        cv2.imshow('contours', img)  # выводим итоговое изображение в окно
    
        cv2.waitKey()
        cv2.destroyAllWindows()
    getContours('blocks.jpg')"""

    """
    def getContoursWithSliders(filename):
        hsv_min = np.array((2, 28, 65), np.uint8)
        hsv_max = np.array((26, 238, 255), np.uint8)
        fn = filename
        img = cv2.imread(fn)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, hsv_min, hsv_max)
        contours0, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        index = 0
        layer = 0
        def update():
            vis = img.copy()
            cv2.drawContours(vis, contours0, index, (255, 0, 0), 2, cv2.LINE_AA, hierarchy, layer)
            cv2.imshow('contours', vis)
        def update_index(v):
            global index
            index = v - 1
            update()
        def update_layer(v):
            global layer
            layer = v
            update()
        update_index(0)
        update_layer(0)
        cv2.createTrackbar("contour", "contours", 0, 7, update_index)
        cv2.createTrackbar("layers", "contours", 0, 7, update_layer)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    getContoursWithSliders('4fig.png')
    """
    def updateCurrentImage(self, file):
        cv2.imwrite('currentFile.png', file)


    def contour(filename, threshold=150, minW = 0, minH = 0):
        image = cv2.imread(filename)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        # draw contours on the original image
        image_copy = image.copy()
        #cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        # see the results
        for i in range(len(contours)):
            cnt = contours[i]
            x,y,w,h = cv2.boundingRect(cnt)
            if (minH < h and minW < w):
                if hierarchy[0, i, 3] == 0:
                    cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    ImageProcessor.cropImage(filename, x, y, w, h, (str(x)+'_'+str(y)+'_'+str(w)+'_'+str(h)+'.png'))
        #cv2.imshow('Bin', thresh)
        cv2.imshow('Rect contour, thresh=old'+ str(threshold) + '|minH='+str(minH)+'|minW='+str(minW), image_copy)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    def cropImage(filename, x, y, w, h, croppedFileName='default.png'):
        img = cv2.imread(filename)
        cropped_image = img[y:y+h, x:x+w]
        cv2.imwrite(str('./crops/'+croppedFileName), cropped_image)

    #contour('4fig.png', 242, 100, 100)


    def showImage(filename):
        img = cv2.imread(filename)
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    """def drawLines(filename, angle, amount):
        img = cv2.imread(filename)
        height, width = img.shape[:2]

        start_p = [0, 0]
        end_p = [0, height]
        color = (255, 0, 0)
        thickness = 1
        step = width // amount
        for i in range(amount):
            start_p = [0+i*step, 0]
            end_p = [0+i*step, height]
            img = cv2.line(img, start_p, end_p, color, thickness)


        cv2.imshow('Lines', img)
        cv2.waitKey(0)"""


    def slope(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        if x2 != x1:
            return ((y2 - y1) / (x2 - x1))
        else:
            return 'NA'


    def drawLines(filename, stripeThreshold, amount):
        image = cv2.imread(filename)
        step = image.shape[1] // amount
        stripesCounter = 0
        print('---------------------------------------------------------\nNew trajectory:')
        for i in range(amount+1):
            p1 = [0+step*i, 0]
            p2 = [0+step*i, image.shape[1]]
            x1, y1 = p1
            x2, y2 = p2
            ### finding slope
            m = ImageProcessor.slope(p1, p2)
            ### getting image shape
            h, w = image.shape[:2]
            if m != 'NA':
                ### here we are essentially extending the line to x=0 and x=width
                ### and calculating the y associated with it
                ##starting point
                px = 0
                py = -(x1 - 0) * m + y1
                ##ending point
                qx = w
                qy = -(x2 - w) * m + y2
            else:
                ### if slope is zero, draw a line with x=x1 and y=0 and y=height
                px, py = x1, 0
                qx, qy = x1, h


            cropped_image = image[0:p2[1], p1[0]:p1[0]+step]
            if (len(cropped_image) > 0 and len(cropped_image[1]) > 0 and len(cropped_image[0]) > 0):
                redPoint1, redPoint2 = ImageProcessor.getSubstripe(cropped_image, [px, py], [qx, qy], step)
                if ImageProcessor.checkStipe(cropped_image, stripeThreshold, step=1, startH=redPoint1[0], stopH=redPoint2[0]):
                    ImageProcessor.writeTrajectory(redPoint1, redPoint2, step, i)
                    stripesCounter += 1
                    cv2.line(image, (step * i, redPoint1[0]), (step * i, redPoint2[0]), (0, 255, 0), 1)
                    #print(i, redPoint1[0], redPoint2[0])
                    cv2.line(image, (step * (i+1), redPoint1[0]), (step * (i+1), redPoint2[0]), (0, 255, 0), 1)
                    #cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), (0, 255, 0), 2)
                    #cv2.line(image, (int(px)+step, int(py)), (int(qx)+step, int(qy)), (0, 255, 0), 2)
        print('Total stripes='+str(stripesCounter)+'|Threshold=' + str(stripeThreshold)+'\n---------------------------------------------------------')
        #cv2.resize(image, (image.shape[0]*2, image.shape[1]*2))
        #image = cv2.copyMakeBorder(image, top=0, bottom=100, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=[10,10,10])
        #image = cv2.putText(image, '5', [image.shape[0]//2, image.shape[1]//2 + 50], cv2.FONT_HERSHEY_SIMPLEX, 10, 0.01, cv2.LINE_AA)

        #cv2.imshow('Threshold=' + str(stripeThreshold) + '|Total=' + str(stripesCounter), image)
        cv2.imshow(str(stripesCounter), image)
        #cv2.waitKey(0)


    def checkStipe(img, threshold=50, step=1, startH=0, stopH=None):
        if stopH == None:
            stopH = len(img)
        #cv2.imshow('File', img)
        #cv2.waitKey(0)
        counter = 0
        for i in range(startH, stopH, step):
            for j in range(0, len(img[i]), step):
                needToCount = False
                if not (img[i][j][0] == 0 and img[i][j][1] == 255 and img[i][j][2] == 0):
                    for k in range(3):
                        if img[i][j][k] < 254:
                            needToCount = True
                if needToCount:
                    counter += 1
        #print(counter)
        #print((len(img)*len(img[0])))
        if stopH==startH:
            return False

        if counter/((stopH-startH)*len(img[0])) > threshold / 100:
            return True
        else:
            return False

    def getSubstripe(img, p1, p2, step, threshold=250):
        startH = [0,0]
        stopH = [0,0]
        startFound = False
        stopFound = False
        """for i in range(0, len(img[0])):
            for j in range(0, len(img)):
                if not (img[j][i][0] == 0 and img[j][i][1] == 255 and img[j][i][2] == 0):
                    for k in range(3):
                        if img[j][i][k] < 250:
                            startH = [i, j]
                            #print(i,j)
                            break"""
        #cv2.imshow('strip', img)
        for i in range(0, len(img)):
            if not startFound:
                for j in range(len(img[0])-1,-1,-1):
                    if not startFound:
                        if not (img[i][j][0] == 0 and img[i][j][1] == 255 and img[i][j][2] == 0):
                            for k in range(3):
                                if img[i][j][k] < 250:
                                    startH = [i, j]
                                    startFound = True
                                    break
        for i in range(len(img)-1, -1, -1):
            if not stopFound:
                for j in range(len(img[0])-1, -1, -1):
                    if not stopFound:
                        for k in range(3):
                            if not (img[i][j][0] == 0 and img[i][j][1] == 255 and img[i][j][2] == 0):
                                if img[i][j][k] < 250:
                                    stopH = [i, j]
                                    stopFound = True
                                    break
        #print(startH, stopH)
        #print(p1, p2)
        return startH, stopH


    def writeTrajectory(p1, p2, step, i):
        print('Robot moves from point: [' + str((step+step*i)/2) + ':' + str(p1[0]) + ']__TO point__[' + str((step+step*i)/2) + ':' + str(p2[0]) + ']')




ip = ImageProcessor
from pathlib import Path
import PySimpleGUI as sg
import shutil



sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.

stripesAmount = [sg.Text('Amount of stripes:'), sg.Text(key='stripesAmount')]

layout = [  [sg.Text('Current file:'), sg.Text(key='currentFile')],
            [sg.Input(key='-INPUT-'),
            sg.FileBrowse(file_types=(("PNG", "*.png *.jpg"), ("ALL Files", "*.*"))),
            sg.Button('Open')],
            [sg.Input(default_text=150, key='threshold', visible=False), sg.Text('Threshold(0-255)', visible=False)],
            [sg.Input(default_text=100, key='minH', visible=False), sg.Text('minH', visible=False)],
            [sg.Input(default_text=100, key='minW', visible=False), sg.Text('minW', visible=False)],
            [sg.Button('Split', visible=False)],
            [],
            [sg.Input(default_text=50, key='stripeThreshold', visible=False), sg.Text('stripeThreshold', visible=False)],
            [sg.Input(default_text=10, key='amount', visible=False), sg.Text('amount', visible=False)],
            [sg.Button('Draw lines', visible=False)],
            #stripesAmount,
            [sg.Input(default_text='0', key='rotatingAngle', visible=False), sg.Text('Angle', visible=False)],
            [sg.Button('Rotate', visible=False)],
            [sg.Button('Cancel')]
            ]
# Create the Window
window = sg.Window('Window Title', layout)
window.finalize()
window.set_title('GUI')
window.size = [500, 500]
filename = ''
# Event Loop to process "events" and get the "values" of the inputs


while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    filePath = values['-INPUT-']
    threshold = int(values['threshold'])
    minW = int(values['minW'])
    minH = int(values['minH'])
    stripeThreshold = int(values['stripeThreshold'])
    amount = int(values['amount'])
    rotatingAngle = int(values['rotatingAngle'])
    if event == 'Open':
        if Path(filePath).is_file():
            shutil.copyfile(filePath, ('./Sources/' + Path(filePath).name))
            filename = './Sources/' + Path(filePath).name
            for i in layout:
                for j in i:
                    j.update(visible=True)
                    if (j.key == 'currentFile'):
                        j.update(filename)

    if event == 'Split':
        print(filename)
        ip.contour(filename, threshold, minW, minH)

    if event == 'Draw lines':
        ip.drawLines(filename, stripeThreshold, amount)
        #ip.drawLines(filename, angle, amount)

    if event == 'Rotate':
        ImageProcessor.rotate_image(filename, rotatingAngle)
        """print(threshold)

        image = cv2.imread(filename)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow('Grayscale', img_gray)
        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        # draw contours on the original image
        image_copy = image.copy()
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        # see the results

        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            if (minH < h and minW < w):
                if hierarchy[0, i, 3] == 0:
                    cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    ImageProcessor.cropImage(filename, x, y, w, h,
                                             (str(x) + '_' + str(y) + '_' + str(w) + '_' + str(h) + '.png'))
        #cv2.imshow('Bin'+str(threshold), thresh)
        cv2.imshow('Contours ' + str(threshold), image_copy)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()"""


window.close()