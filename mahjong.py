import io
import picamera
import cv2
import numpy
from time import sleep
from imutils import build_montages
from imutils import paths
import glob
import os
import os.path

cv2.ocl.setUseOpenCL(False)

examples = []
sift = cv2.xfeatures2d.SIFT_create()
print "Loading examples..."
for file in os.listdir("./examples"):
 exampleImage = cv2.imread("./examples/" + file,0)  
 kp, des = sift.detectAndCompute(exampleImage, None)
 examples.append((file.split(".", 1)[0].split("_", 1)[0], kp, des))
print str(len(examples)) + " examples loaded."

matcher = cv2.BFMatcher()

while True:
        a = raw_input("enter to start scanning")

        # Get image from raspberry pi camera
	stream = io.BytesIO()
        width = 640*2
        height = 480*2
	with picamera.PiCamera() as camera:
	 camera.resolution = (width, height)
         camera.iso = 400
         camera.awb_mode = "auto"
         camera.contrast = 0
         camera.sharpness = 50
	 camera.capture(stream, format='jpeg')

	buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

	image = cv2.imdecode(buff, 1)

        # Assuming that tiles are in lower part of image
        image = image[(height/2):height, 0:width]

        # Grayscale and blur, blur is required to make the edge detection work
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)

        # Threshold to simplify images
        ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
  
        # Edge detection
        edges = cv2.Canny(thresh, 80, 255, apertureSize = 3)
	cv2.imwrite('gray.jpg',gray)
	cv2.imwrite('thresh.jpg',thresh)
	cv2.imwrite('edges.jpg',edges)

        # Now find the contours of the rectangles
	im2, cnts, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_index = 1
	small_images = []
        tile_matches = []
	for c in cnts:
          peri = cv2.arcLength(c, True)
          approx = cv2.approxPolyDP(c, 0.04 * peri, True)
	  area = cv2.contourArea(c)
          if len(approx)>=4 and area > 500:
           rect = cv2.minAreaRect(c)
           box = cv2.boxPoints(rect)
           box = numpy.int0(box)
           cv2.drawContours(image, [box],0, (0,0,255), 2)

	   W = rect[1][0]
	   H = rect[1][1]
	   Xs = [i[0] for i in box]
	   Ys = [i[1] for i in box]
	   x1 = min(Xs)
	   x2 = max(Xs)
	   y1 = min(Ys)
	   y2 = max(Ys)
	   angle = rect[2]
	   if angle < -45:
	    angle += 90
	   center = ((x1+x2)/2,(y1+y2)/2)
           size = (x2-x1, y2-y1)
	   M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
           cropped = cv2.getRectSubPix(image, size, center)
           cropped = cv2.warpAffine(cropped, M, size)
           croppedW = H if H > W else W
           croppedH = H if H < W else W
           croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW),int(croppedH)), (size[0]/2, size[1]/2))
           croppedRotated = cv2.resize(croppedRotated, (64, 64))
           small_images.append(croppedRotated)
	   gray_small = cv2.cvtColor(croppedRotated,cv2.COLOR_BGR2GRAY)
           kp, des = sift.detectAndCompute(gray_small, None)
           best_count = 0
           best = ""
	   for file, ex_kp, ex_des in examples:
	     matches = matcher.knnMatch(ex_des, des, k = 2)
             goodCount = 0
             if matches != None: 
       	      for m,n in matches:
               if m.distance < 0.7*n.distance:
                goodCount=goodCount+1
             if best_count < goodCount:
              best_count = goodCount
              best = file

	   cv2.imwrite('cropped'+str(image_index)+'.jpg',croppedRotated)
           image_index=image_index+1
	   cv2.drawContours(image, [c], -1, (0,255,0), 2) 
           M = cv2.moments(c)
           cX = int((M["m10"]/M["m00"]))
           cY = int((M["m01"]/M["m00"]))
           if best_count >= 5:
            tile_matches.append(best)
            cv2.putText(image, best, (cX,cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
        print str(len(small_images)) + " tiles found"
        for match in tile_matches:
         print match
         
        montage = build_montages(small_images, (320,200), (10, 1))
	for m in montage:
	 cv2.imwrite('montage.jpg',m)
	cv2.imwrite('output.jpg',image)
