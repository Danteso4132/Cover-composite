

import pylab as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
try:
    from PIL import Image
except ImportError:
    import Image
import cv2
import pylab as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
class ImageProcessor:
    try:
        from PIL import Image
    except ImportError:
        import Image
    import cv2
    import numpy as np




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
    def rotate_image(filename, angle):
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
        #rotated_mat = cv2.resize(rotated_mat, (width, height))
        print(width, height)
        cv2.imshow('original Image', mat)
        cv2.namedWindow('Rotated Image', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Rotated Image', width, height)


        cv2.imshow('Rotated Image', rotated_mat)
        cv2.imwrite('rotated.png', rotated_mat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return rotated_mat

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
        cv2.imshow('Bin', thresh)
        cv2.imshow('None approximation', image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cropImage(filename, x, y, w, h, croppedFileName):
        img = cv2.imread(filename)
        cropped_image = img[y:y+h, x:x+w]
        cv2.imwrite(str('./crops/'+croppedFileName), cropped_image)

    #contour('4fig.png', 242, 100, 100)


