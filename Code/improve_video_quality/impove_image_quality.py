import cv2
import numpy as np
import traceback
import copy


def equalize_light(image, limit=12.0, grid=(2,2), gray=False):
    if (len(image.shape) == 2):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = True
    
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    cl = clahe.apply(l)
    #cl = cv2.equalizeHist(l)
    limg = cv2.merge((cl,a,b))

    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if gray: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return np.uint8(image)


def gamma_correction(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

cap = cv2.VideoCapture('../media/problem1_data/Night Drive-2689.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("problem1_image_quality.mp4", fourcc, 20.0, (1280, 720))
while (True):
    try:
        
        ret_val, image = cap.read() #True, image #
        if not ret_val:
            #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
        else:
            image = cv2.resize(image,(int(500),int(500)))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            im_g = copy.deepcopy(image)
            eq = copy.deepcopy(image)
            eq = equalize_light(eq)
            
            eq_g = copy.deepcopy(eq)
            eq_g = gamma_correction(eq_g, 1.5)
            im_g = gamma_correction(im_g, 2.6)
            
            cv2.imshow('Original', image)
            cv2.imshow('Contrast Limited Adaptive Histogram Equalization - CLAHE', eq)
            cv2.imshow('Gamma Correction', im_g)
            cv2.imshow('CLAHE and Gamma Correction', eq_g)
            out.write(eq_g)
            
        # Esc key to stop the processing
        if cv2.waitKey(1) == 27:
            break
    
    except Exception: 
        traceback.print_exc()
        break


cap.release()
cv2.destroyAllWindows()