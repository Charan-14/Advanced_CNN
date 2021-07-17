####################################################################################################################################################################################
#INTERNSHIP 2019 DECEMBER
####################################################################################################################################################################################
from PyQt5.QtCore import QDate, QTime, Qt, pyqtSlot
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QToolTip, QMainWindow, QLineEdit, QLabel, QVBoxLayout
from PyQt5.QtGui import QIcon, QFont, QPixmap
import os
import time
import cv2 as cv
import numpy as np
import datetime
from tinydb import TinyDB, Query
import pytesseract
import pytesseract
import xlwt
from xlwt import Workbook
from collections import Counter
import statistics
import math

previous2_good = 0
previous_good = 0
date_today = datetime.date.today()
directory = r'C:\Users\matlab.pc\Desktop\Toyota_Tests_Results'
os.chdir(directory)
db_time_product = TinyDB('Testing.json')
max_count = 0
DateSearch = Query()
list_of_today = db_time_product.search(DateSearch.Date == str(date_today))
if len(list_of_today)!= 0:
    for i in list_of_today:
        if i['Count']>max_count:
            max_count = i['Count']
cap3 = cv.VideoCapture(1) #Barcode side
cap = cv.VideoCapture(2)    #Top view
cap1 = cv.VideoCapture(0)  #Washer Side
cap2 = cv.VideoCapture(3)
print("GUI started")
cap.set(cv.CAP_PROP_AUTOFOCUS, False)
cap.set(cv.CAP_PROP_FOCUS, 22)
circles = []
circle_found = 0
radius_circles = []
terminals_found_list = []
error_terminals = 1
terminals_found = 0
total_testing_today = 0
wb = Workbook()
sheet1 = wb.add_sheet("Barcodes",cell_overwrite_ok=True)
date_and_product = []
average_distance_length= []
hose_pipe_detected = 0
yellow_dot_found = 0
def average(lst):
    if len(lst) != 0:
        return sum(lst)/len(lst)
    else:
        return (0)

def rotate_image(image):
    image_center = tuple(np.array(image.shape[1::-1])/2)
    rot_mat = cv.getRotationMatrix2D(image_center, 180, 1)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags = cv.INTER_LINEAR)
    return result

def yellow_dot():
    global yellow_dot_found
    global average_distance_length
    _, frame = cap3.read()
    original = frame.copy()
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsv_y = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_y = np.array([12, 98, 120 ])#lower_y = np.array([8, 122, 115])
    upper_y = np.array([38, 255, 255])#upper_y = np.array([47, 172, 151])
    mask_y = cv.inRange(hsv_y, lower_y, upper_y)
    #cv.imshow("Maskin Yellow", mask_y)
    res_y = cv.bitwise_and(frame, frame, mask=mask_y)
    gray_y = cv.cvtColor(res_y, cv.COLOR_BGR2GRAY)
    contour_y, _ = cv.findContours(gray_y, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contour_y)!=0:
        #cv.drawContours(gray_y, contour_y, -1, 255, 3)
        cmaxy = max(contour_y, key = cv.contourArea)
        
        if cv.contourArea(cmaxy)>100 and cv.contourArea(cmaxy)<1000 :
            

            (xy, yy, wy, hy) = cv.boundingRect(cmaxy)
            if xy < 320:
                yellow_dot_found += 1
                print("Yellow dot found")
                cv.rectangle(frame, (xy,yy), (xy+wy, yy+hy), (0, 255, 0), 2)

                frame_tailpipe = frame[yy-120:yy+180,xy+295:xy+440]
                if xy > 340:
                    pass
                else:
                    if len(frame_tailpipe)==0:
                        pass
                    else:
                        
                        gray_tp = cv.cvtColor(frame_tailpipe, cv.COLOR_BGR2GRAY)
                        retthr, thr1 = cv.threshold(gray_tp, 113, 1000, cv.THRESH_BINARY_INV)
                        contour_tp, _ = cv.findContours(thr1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                        frame_tp = frame_tailpipe.copy()
                        
                        if len(contour_tp)!=0:
                            cv.drawContours(gray_tp, contour_tp, -1, 255, 3)
                            cmax_tp = max(contour_tp, key = cv.contourArea)
                            if cv.contourArea(cmax_tp)>0:
                                (xt, yt, wt, ht) = cv.boundingRect(cmax_tp)
                                cv.rectangle(frame_tp, (xt,yt), (xt+wt, yt+ht), (0, 0, 255), 2)
                                dist = math.sqrt((xt - (xt+wt))**2 + (yt-(yt+ht))**2)
                                cv.line(frame_tp, (xt,yt), (xt+wt, yt+ht), (0, 255, 0), 2, cv.LINE_AA)
                                average_distance_length.append(dist)  
            
        else:
            yellow_dot_found -= 1
            print("No yellow dot detected")

def hosepipe():
    _, frame = cap3.read()
    original = frame.copy()
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    global hose_pipe_detected
    hsv_g = cv.cvtColor(original, cv.COLOR_BGR2HSV)
    lower_g = np.array([80, 15, 12])
    upper_g = np.array([140, 70, 55])
    mask_g = cv.inRange(hsv_g, lower_g, upper_g)
    res_g = cv.bitwise_and(original, original, mask=mask_g)
    #cv.imshow("HOSE PIPE show", res_g)
    gray_g = cv.cvtColor(res_g, cv.COLOR_BGR2GRAY)
##    cv.imshow('3rd Frame', frame)
##    cv.moveWindow("3rd Frame", 1250, 500)
    contour_g, _ = cv.findContours(gray_g, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contour_g)!=0:
        print("Contour found")
        cv.drawContours(gray_g, contour_g, -1, 255, 3)
        cmaxg = max(contour_g, key = cv.contourArea)
        

        if cv.contourArea(cmaxg)>500  :
            (xg, yg, wg, hg) = cv.boundingRect(cmaxg)
            if xg< 200 and (xg+ wg)< 250:
                
        
                cv.rectangle(frame, (xg,yg), (xg+wg, yg+hg), (0, 255, 0), 2)
                hose_pipe_detected += 10
                print("Hose pipe Detected")
        else:
            hose_pipe_detected -= 1
            print("Hose pipe Missing")


def draw_polygon(contours, frame, gray):
    largest_area = 0
    (xg_l, yg_l, wg_l, hg_l) = (0,0,0,0)
    k = 0
    global washer_detected
    screw_x = 0
    contours_that_satisfy= []
    
    if len(contours) != 0:
        for i in range(len(contours)):
            if cv.contourArea(contours[i]) > 90 and cv.contourArea (contours[i]) < 700:
                
                perimeter = cv.arcLength(contours[i], True)
                epsilon = 0.01* perimeter
                approx = cv.approxPolyDP(contours[i], epsilon, True)
                if len(approx) > 3:
                    (xg, yg, wg, hg) = cv.boundingRect(contours[i])
                    if (yg> 350) and yg < 390 and xg> 330 and xg < 390 and  wg > 20 and wg< 60 and hg< 50 and hg> 10:
                        flag = 0
                        if wg* hg > largest_area:
                            largest_area = wg*hg
                            contour_with_largest_area = contours[i]
                            (xg_l, yg_l, wg_l, hg_l) = cv.boundingRect(contours[i])
                            screw_x = xg_l
                            print("Screw Found")
                        cv.rectangle(original, (xg_l, yg_l), (xg_l+wg_l, yg_l+hg_l), (0, 0, 255), 2)
                        
                        washer_ROI = gray[yg_l-10: yg_l+2, xg_l-10:xg_l+wg_l+10]
                        
                        th_washer = cv.adaptiveThreshold(washer_ROI,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 1)
                        
                        contours_washer, _ = cv.findContours(th_washer, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                        if len(contours_washer) != 0:
                            for i in range(len(contours_washer)):
                                perimeter = cv.arcLength(contours_washer[i], True)
                                epsilon = 0.01* perimeter
                                approx = cv.approxPolyDP(contours_washer[i], epsilon, True)
                                if len(approx) > 3 and len(approx) < 15:
                                    (xg, yg, wg, hg) = cv.boundingRect(contours_washer[i])
                                    if wg>30 and wg<60 and hg< 10:
                                        cv.rectangle(washer_ROI, (xg, yg), (xg+wg, yg+hg), (0, 255, 255), 2)
                                        print("Washer found")
                                        washer_detected += 10
                                        flag = 1

                        if flag == 0:
                                print("Washer doesn't exist")
                                washer_detected -= 1
                        print("HereNow")
                        
                        k = cv.waitKey(1) & 0xFF
                        if k == 27:
                            break
    print("Here1")
    return(screw_x)    
            

            
def topframe():
    global circle_found
    global error_terminals
    global terminals_found
    global total_testing_today
    global frame
    global original
    global img
    global hose_pipe_detected
    global yellow_dot_found
    global average_distance_length
    global washer_detected
    start_time = int(round(time.time()*1000))
    radius_circles = []
    terminals_found_list = []
    circle_found = 0
    error_terminals = 1
    terminals_found = 0
    date_and_product = []
    average_distance_length= []
    hose_pipe_detected = 0
    yellow_dot_found = 0
    washer_detected = 0
    filter_distance = []
    difference = []
    red_gap = []
    while True:
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FOCUS, 25)
            ret, frame = cap.read()
            frame_top = frame.copy()
            original = frame.copy()
            frame_top = cv.resize(frame_top, (320,240))
            cv.imshow("LIVE", frame_top)
            cv.moveWindow("LIVE", 1250, 20)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            edges = cv.Canny(gray, 50, 200)
            ret, thresh = cv.threshold(gray, 70, 255, cv.THRESH_BINARY)
            th3 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            lower_blue1 = np.array([90,68,33])
            upper_blue1 = np.array([140,246,220])
            mask = cv.inRange(hsv, lower_blue1, upper_blue1)
            blue_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
            res_blue = cv.bitwise_and(frame, frame, mask = blue_mask)
            med = cv.medianBlur(gray, 3)
            th5 = cv.adaptiveThreshold(med, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)
            contours_blue, _ = cv.findContours(blue_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            
            if len(contours_blue)!= 0:
                cmax = max(contours_blue, key = cv.contourArea)
                if cv.contourArea(cmax) > 500: 
                    (xg, yg, wg, hg) = cv.boundingRect(cmax)
                    (x, y, w, h) = cv.boundingRect(cmax)
                    xg = xg + round(6*wg/5)
                    yg = yg + round(hg/3)
                    wg = round(wg/5)
                    hg = round(hg/3)
                    cv.rectangle(original, (xg, yg), (xg+wg, yg+hg), (255, 0, 255), 2)
                    ROI_adaptive = th5[yg:yg+hg, xg:xg+wg]
                    ROI_COPY = frame[yg:yg+hg, xg:xg+wg]
                    circles = cv.HoughCircles(ROI_adaptive, cv.HOUGH_GRADIENT, 1, 20, param1=5,param2=15,minRadius=0,maxRadius=9)
                    if circles is not None:
                        circles = np.round(circles[0, :]).astype("int")
                        if len(circles) == 1:
                            for (xc,yc,rc) in circles:
                                radius_circles.append(rc)
                                circle_found += 1
                    
                    ROI = original[y:y+h,x:x+w]
                    ROI_copy = ROI.copy()
                    ROI_left = original[round(y+2.5*h/10):round(y+8*h/10),round(x+2*w/10):round(x+4*w/10)]
                    ROI_left_copy = ROI_left.copy()
                    ROI_left_copy_copy = ROI_left.copy()
                    lower_white = np.array([0, 45, 0])
                    upper_white = np.array([180, 255, 255])
                    hsv_white = cv.cvtColor(ROI_left_copy, cv.COLOR_BGR2HSV)
                    mask_white = cv.inRange(hsv_white, lower_white, upper_white)
                    gray_ROI = cv.cvtColor(ROI_left, cv.COLOR_BGR2GRAY)
                    #cv.imshow("gray_ROI", gray_ROI)
                    ret, thresh_ROI = cv.threshold(gray_ROI, 114,255, cv.THRESH_BINARY)
                    th4 = cv.adaptiveThreshold(gray_ROI, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 2)
                    contours_th4, _ = cv.findContours(thresh_ROI, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    if len(contours_th4) != 0:  
                        for i in range(len(contours_th4)):
                            (xg, yg, wg, hg) = cv.boundingRect(contours_th4[i])
                            cv.rectangle(ROI_left_copy, (xg, yg), (xg+wg, yg+hg), (0,200,255), 1)
                         
                    contours_mask_white_1, _ = cv.findContours(mask_white, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    contours_mask_white = []
                    k = 1
                    for cont in range(len(contours_mask_white_1)):
                        useful_contour = 0
                        
                        if cv.contourArea(contours_mask_white_1[cont]) > 50 and cv.contourArea(contours_mask_white_1[cont]) < 150:
                            (x1,y1,w1,h1) = cv.boundingRect(contours_mask_white_1[cont])
                            _, width = mask_white.shape
                            
                            if x1>round(width/10) and x1< round(5*width/10) and w1< round(3*width/10):
                                if k == 1:
                                    first_x = x1
                                    k += 1
                                    contours_mask_white.append(contours_mask_white_1[cont])
                                else:
                                    print("DIFFERENCE IN pin: ", x1-first_x)
                                    if x1 < (first_x +5) and x1> (first_x-5):
                                        contours_mask_white.append(contours_mask_white_1[cont])
                                
                    if len(contours_th4) == len(contours_mask_white) == 2:
                        error_terminals = 2
                        terminals_found_list.append(error_terminals)
                        terminals_found_list.append(error_terminals)
                        print("Print the number of pins are: ", len(contours_th4))
                    else:
                        error_terminals = 1
                        terminals_found_list.append(error_terminals)
        
        if cap3.isOpened():
            _, frame = cap3.read()
            frame_third = frame.copy()
            frame_third = cv.resize(frame_third, (320, 240))
            cv.imshow('3rd Frame', frame_third)
            
            cv.moveWindow("3rd Frame", 1600, 500)
            original = frame.copy()
            img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            yellow_dot()
            hosepipe()

        if cap2.isOpened():
            _, frame = cap2.read()
            frame_second = frame.copy()
            frame_second = cv.resize(frame_second, (320,240))
            cv.imshow('2nd Frame', frame_second)
            cv.moveWindow("2nd Frame", 1600, 20)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)       
            lower_pipe = np.array([81, 0, 39 ])
            upper_pipe = np.array([97 , 70 , 77])
            kernel = np.ones((3,3),np.uint8)
            mask_pipe = cv.inRange(hsv, lower_pipe, upper_pipe)
            res_pipe = cv.bitwise_and(frame, frame, mask=mask_pipe)
            res_pipe = cv.morphologyEx(res_pipe, cv.MORPH_CLOSE, kernel, iterations=3)
            res_pipe = cv.dilate(res_pipe, kernel, iterations=1)
            pipe_roi = res_pipe[90:480, 110:640]
            frame_pipe = frame[90:480, 110:640]

            lower_red = np.array([0, 102, 33])
            upper_red = np.array([27, 255, 120])

            mask_red = cv.inRange(hsv, lower_red, upper_red)
            res_red = cv.bitwise_and(frame, frame, mask=mask_red)
            res_red = res_red[0:480, 110:640]
            frame_redbase = frame[0:480, 110:640]
            gray_r = cv.cvtColor(res_red, cv.COLOR_BGR2GRAY)

            
            gray_pipe = cv.cvtColor(pipe_roi, cv.COLOR_BGR2GRAY)
            contour_pipe, _ = cv.findContours(gray_pipe, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            contour_pipe = sorted(contour_pipe, key=cv.contourArea, reverse=True)[:2]

            
            contour_red, _ = cv.findContours(gray_r, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            #print("Totla contours in red", len(contour_red))
            contour_red = sorted(contour_red, key=cv.contourArea, reverse=True)[:1]
            if len(contour_pipe)!=0:    
                for contour in contour_pipe:
                    (xp, yp, wp, hp) = cv.boundingRect(contour)
                    cv.rectangle(frame_pipe, (xp,yp), (xp+wp, yp+hp), (0, 0, 255), 2)
                
                (x1,y1,w1,h1) = cv.boundingRect(contour_pipe[0])
                (x2,y2,w2,h2) = cv.boundingRect(contour_pipe[1])    
                align = abs(h1-h2)
                difference.append(align)
                
                if len(contour_red)!=0:
                    for contour in contour_red:
                        (xr, yr, wr, hr) = cv.boundingRect(contour)
                        gap = abs((y2+h2)-(yr+hr))
                        print("gap - " , gap)
                        red_gap.append(gap)
                #hosepipe_detected += 10
   
            else:
                print("Hosepipe not detected")
                #hosepipe_detected -= 1

        if cap1.isOpened():
            _, frame = cap1.read()
            aaa = frame.copy()
            frame_first = frame.copy()
            frame_first = cv.resize(frame_first, (320,240))
            cv.imshow('1st Frame', frame_first)
            cv.moveWindow("1st Frame", 1250, 500)
            original = frame.copy()
            frame1 = frame.copy()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            kernel = np.ones((3,3), np.float32)/9
            dst = cv.filter2D(gray, -1, kernel)
            th6 = cv.adaptiveThreshold(dst,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 1)
            contours_box_blur, _ = cv.findContours(th6, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            print("Here")
            screw_x = draw_polygon(contours_box_blur, th6, gray)
            print("Here")
            #cap1.relese()
            _, frame = cap1.read()
            original = frame.copy()
            frame1 = frame.copy()
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            lower = np.array([25,0,0])
            upper = np.array([63,255,20])
            mask = cv.inRange(hsv, lower, upper)
            blackmask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
            res_black = cv.bitwise_and(frame, frame, mask=blackmask)
            gray_black = cv.cvtColor(res_black, cv.COLOR_BGR2GRAY)
            contours_black, _ = cv.findContours(gray_black, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            
            if len(contours_black) != 0:
                cv.drawContours(original, contours_black, -1, (0,255, 0), 2)
                cmax = max(contours_black, key = cv.contourArea)
                if cv.contourArea(cmax) > 600:
                    perimeter = cv.arcLength(cmax, True)
                    epsilon = 0.005* perimeter
                    approx = cv.approxPolyDP(cmax, epsilon, True)
                    if len(approx) > 3:
                        (xg, yg, wg, hg) = cv.boundingRect(cmax)
                        if (yg> 150) and yg < 330 and xg> 50 and xg < 340 and (xg+wg) < 340 and wg< 200:
                            filter_found = 1
                            cv.rectangle(original, (xg, yg), (xg+wg, yg+hg), (0, 255, 255), 2)
                            filter_x_edge = xg+wg
                        else:
                            filter_found = 0
            else:
                filter_found = 0

            if filter_found == 0:
                print("Filter not found")
            else:
                
                print("Filter found")
                print("Distance of filter ", filter_x_edge)
                if screw_x != 0:
                    distance= screw_x - filter_x_edge
                    print("Distance of filter and center is: ", distance)
                    filter_distance.append(distance)
            
            
            

        current_time = int(round(time.time()*1000))
        diff_time = current_time - start_time
        
        k = cv.waitKey(1) & 0xFF

        if diff_time > 7000:
            max_count = 0
            DateSearch = Query()
            list_of_today = db_time_product.search(DateSearch.Date == str(date_today))
            if len(list_of_today)!= 0:
                for i in list_of_today:
                    if i['Count']>max_count:
                        max_count = i['Count']
            max_count += 1
            db_time_product.insert(({'Date': str(date_today), 'Count':max_count}))
            label_time.setText(str(date_today))
            date_filename=str(date_today)+ str("-") + str(max_count) + str("-") + str("TOP")
            filename_write = str(date_filename) + str(".jpeg")
            cv.imwrite(filename_write, frame_top)
            date_filename=str(date_today)+ str("-") + str(max_count) + str("-") + str("THIRD")
            filename_write = str(date_filename) + str(".jpeg")
            cv.imwrite(filename_write, frame_third)
            date_filename=str(date_today)+ str("-") + str(max_count) + str("-") + str("FIRST")
            filename_write = str(date_filename) + str(".jpeg")
            cv.imwrite(filename_write, frame_first)
            date_filename=str(date_today)+ str("-") + str(max_count) + str("-") + str("SECOND")
            filename_write = str(date_filename) + str(".jpeg")
            cv.imwrite(filename_write, frame_second)
            
            
            textbox9.setText(str(max_count))
            break
################ TESTING AND VALUE ADJUSTMENT AFTER THIS#########################
    overall_good = 0
    average_radius_of_circle = average(radius_circles)
    error_terminals = round(average(terminals_found_list))
    terminals_found = str(error_terminals)
    length_tail_pipe = round(average(average_distance_length))
    filter_distance = round(average(filter_distance))
    hose_diff = round(average(difference))
    red_gap_diff = round(average(red_gap))

    
    print("FINE TILL HERE")
    if circle_found > 0 and average_radius_of_circle < 7 and average_radius_of_circle > 3 :
        print("We have found the red nosal hole of radius", average_radius_of_circle)
        text = (str(average_radius_of_circle))
        textbox1.setText(text)
        button1.setStyleSheet("background-color: green");
        overall_good += 1
    else:
        print("Problem in the nosal hole", average_radius_of_circle)
        textbox1.setText("Problem in the nosal hole")
        button1.setStyleSheet("background-color: red");
        overall_good -= 1000
        
    if length_tail_pipe > 100 and length_tail_pipe< 350:
        print("We have found the tail pipe of length: ", length_tail_pipe)
        text = str(length_tail_pipe)
        textbox7.setText(text)
        button7.setStyleSheet("background-color: green");
        overall_good += 1
    else:
        print("Problem in tail pipe:", length_tail_pipe)
        textbox7.setText("Problem in the tail pipe")
        button7.setStyleSheet("background-color: red");
        overall_good -= 1000

    if error_terminals == 2:
        print("Terminals found: ", terminals_found)
        text2 = str(terminals_found)
        textbox2.setText(text2)
        button2.setStyleSheet("background-color: green");
        overall_good += 1
    else:
        print("Error in terminals")
        textbox2.setText("Error in terminals")
        button2.setStyleSheet("background-color: red");
        overall_good -= 1000
        
    if hose_pipe_detected > 0:
        print("Hose pipe", hose_pipe_detected)
        textbox5.setText("Hose Pipe Detected")
        print("Hose pipe detected")
        button5.setStyleSheet("background-color: green");
        overall_good += 1
        print("Fine till here")
        print("Hos pipe:", hose_diff)
        if hose_diff< 50:    #HOSE PIPE GAP ADJUSTMENT
            textbox110.setText("Hose Pipe Aligned")
            print("Hose pipe Aligned")
            button110.setStyleSheet("background-color: green");
            overall_good += 1
        else:
            textbox110.setText("Hose Pipe Not Aligned")
            print("Hose pipe Not Aligned")
            button110.setStyleSheet("background-color: red");
            overall_good -= 100
        if red_gap_diff > 90 and red_gap_diff < 137:        # RED GAP ALIGNENT
            textbox111.setText(str(red_gap_diff))
            print("Red Gap Aligned: ", red_gap_diff)
            
            button111.setStyleSheet("background-color: green");
            overall_good += 1
        else:
            textbox111.setText(str(red_gap_diff))
            print("RED GAP Not Aligned: ", red_gap_diff)
            button111.setStyleSheet("background-color: red");
            overall_good -= 100           
        
    else:
        textbox5.setText("Hose Pipe Absent")
        button5.setStyleSheet("background-color: red");
        print("Hose pipe absent")
        textbox110.setText("Hose Pipe Not Aligned")
        print("Hose pipe Not Aligned")
        button110.setStyleSheet("background-color: red");
        textbox111.setText("RED GAP NOT ALIGNMED")
            
        button111.setStyleSheet("background-color: red");
        overall_good -= 100

    if yellow_dot_found> 0:
        textbox8.setText("Yellow Dot found")
        print("Yellow Dot found")
        print("ALL checking done")
        overall_good += 1
        button8.setStyleSheet("background-color: green");
    else:
        textbox8.setText("Yellow Dot Absent")
        print("Yellow Dot absent absent")
        textbox7.setText("Yellow Dot Absent")
        overall_good -= 1000
        button8.setStyleSheet("background-color: red");

    if washer_detected > 0:
        textbox3.setText("Washer Found")
        print("Washer found")
        overall_good += 1
        button3.setStyleSheet("background-color: green");
    else:
        textbox3.setText("Washer NOT Found")
        print("Washer NOT found")
        overall_good -= 100
        button3.setStyleSheet("background-color: red");

    if filter_distance > 85 and filter_distance < 103:   #FILTER GAP ADJUSTMENT
        textbox4.setText(str(filter_distance))
        print("Filter correct")
        print("Filter Distance:", filter_distance)
        overall_good += 1
        button4.setStyleSheet("background-color: green");
    else:
        textbox4.setText("Filter Gap incorrect")
        print("Filter incorrect")
        overall_good -= 100
        print("Filter Distance:", filter_distance)
        button4.setStyleSheet("background-color: red");

    

    if overall_good>0:
        print("GOOD")
        decision_button.setStyleSheet("background-color: green")
        decision_button.setText("GOOD")
        previous_good = 1
    else:
        print("NOT GOOD")
        decision_button.setStyleSheet("background-color: red")
        decision_button.setText("NOT GOOD")
        previous_good = 0


        
def on_start_button():
    global previous_good
    global previous2_good
    w.statusBar().showMessage('Stating the process')
    textbox1.setText("Starting the testing")
    textbox2.setText("Starting the testing")
    textbox3.setText("Starting the testing")
    textbox4.setText("Starting the testing")
    textbox5.setText("Starting the testing")
    textbox7.setText("Starting the testing")
    textbox8.setText("Starting the testing")
    textbox110.setText("Starting the testing")
    textbox111.setText("Starting the testing")
    decision_button.setStyleSheet("background-color: yellow")
    decision_button.setText("CHECKING")
    button1.setStyleSheet("background-color: yellow");
    button2.setStyleSheet("background-color: yellow");
    button3.setStyleSheet("background-color: yellow");
    button4.setStyleSheet("background-color: yellow");
    button5.setStyleSheet("background-color: yellow");
    button7.setStyleSheet("background-color: yellow");
    button8.setStyleSheet("background-color: yellow");
    button110.setStyleSheet("background-color: yellow");
    button111.setStyleSheet("background-color: yellow");
##    if previous2_good == 1:
##        decision_button2.setStyleSheet("background-color: GREEN")
##        decision_button2.setText("GOOD PART")
##    else:
##        decision_button2.setStyleSheet("background-color: RED")
##        decision_button2.setText("NOT GOOD PART")        
    previous2_good = previous_good
    topframe()
     
def on_stop_button():
    w.statusBar().showMessage('Exiting the application')
    cv.destroyAllWindows()
    cap.release()
    cap3.release()
    w.close()

app = QApplication(sys.argv)
w = QMainWindow()
w.setWindowTitle('VISUAL TESTING')
w.setGeometry(0,0,1250,800)
lay = QLabel(w)
lay.resize(200,200)
lay.move(10,10)
pixmap = QPixmap('PVNA_logo.jfif')
lay.setPixmap(pixmap)
button = QPushButton("START", w)
button.setText("START")
button.setToolTip('Click to Start the program')
button.clicked.connect(on_start_button)
button.setStyleSheet("background-color: green");
button.resize(200,100)
button.move(10,250)
stop_button = QPushButton("QUIT", w)
stop_button.clicked.connect(on_stop_button)
stop_button.resize(200,100)
stop_button.setStyleSheet("background-color: red");
stop_button.move(10,400)
w.statusBar()



label_para = QLabel(w)
label_para.move(400, 20)
label_para.setText("<b>PARAMETER</b>")

label_result = QLabel(w)
label_result.move(500, 20)
label_result.setText("<b>RESULT</b>")

label_range = QLabel(w)
label_range.move(720, 20)
label_range.setText("<b>RANGE</b>")

label_time = QLabel(w)
label_time.move(1000, 20)
label_time.setText(str(date_today))

label1 = QLabel(w)
label1.move(400, 50)
label1.setText("<b>RADIUS:</b>")
textbox1 = QLineEdit(w)
textbox1.move(500, 50)
textbox1.resize(150, 20)
textbox1.setText("Results")

label11 = QLabel(w)
label11.move(720, 45)
label11.setText("3<x<7")

label2 = QLabel(w)
label2.move(400, 100)
label2.setText("<b>TERMINALS:</b>")
textbox2 = QLineEdit(w)
textbox2.move(500, 100)
textbox2.resize(150, 20)
textbox2.setText("Results")

button1 = QPushButton('', w)
button1.resize(20,20)
button1.move(670, 50)
button1.setStyleSheet("background-color: yellow");

button2 = QPushButton('', w)
button2.resize(20,20)
button2.move(670, 100)
button2.setStyleSheet("background-color: yellow");

label3 = QLabel(w)
label3.move(400, 150)
label3.setText("<b>WASHER:</b>")
textbox3 = QLineEdit(w)
textbox3.move(500, 150)
textbox3.resize(150, 20)
textbox3.setText("Results")
button3 = QPushButton('', w)
button3.resize(20,20)
button3.move(670, 150)
button3.setStyleSheet("background-color: yellow");

label4 = QLabel(w)
label4.move(400, 200)
label4.setText("<b>FILTER</b>")
textbox4 = QLineEdit(w)
textbox4.move(500, 200)
textbox4.resize(150, 20)
textbox4.setText("Results")
button4 = QPushButton('', w)
button4.resize(20,20)
button4.move(670, 200)
button4.setStyleSheet("background-color: yellow");
label41 = QLabel(w)
label41.move(720, 200)
label41.setText("85<x<103")

label5 = QLabel(w)
label5.move(400, 250)
label5.setText("<b>HOSE PIPE</b>")
textbox5 = QLineEdit(w)
textbox5.move(500, 250)
textbox5.resize(150, 20)
textbox5.setText("Results")
button5 = QPushButton('', w)
button5.resize(20,20)
button5.move(670, 250)
button5.setStyleSheet("background-color: yellow");

label6 = QLabel(w)
label6.move(400, 300)
label6.setText("<b>BARCODE</b>")
textbox6 = QLineEdit(w)
textbox6.move(500, 300)
textbox6.resize(150, 20)
textbox6.setText("Barcode scanner not Found")
button6 = QPushButton('', w)
button6.resize(20,20)
button6.move(670, 300)
button6.setStyleSheet("background-color: yellow");

label7 = QLabel(w)
label7.move(400, 350)
label7.setText("<b>TAIL PIPE</b>")
textbox7 = QLineEdit(w)
textbox7.move(500, 350)
textbox7.resize(150, 20)
textbox7.setText("Results")
button7 = QPushButton('', w)
button7.resize(20,20)
button7.move(670, 350)
button7.setStyleSheet("background-color: yellow");

label71 = QLabel(w)
label71.move(720, 345)
label71.setText("100<x<350")

label75 = QLabel(w)
label75.move(720, 495)
label75.setText("90<x<137")

label8 = QLabel(w)
label8.move(400, 400)
label8.setText("<b>YELLOW DOTS</b>")
textbox8 = QLineEdit(w)
textbox8.move(500, 400)
textbox8.resize(150, 20)
textbox8.setText("Results")
button8 = QPushButton('', w)
button8.resize(20,20)
button8.move(670, 400)
button8.setStyleSheet("background-color: yellow");

label110 = QLabel(w)
label110.resize(200, 20)
label110.move(350, 450)
label110.setText("<b>HOSE PIPE ALIGNMENT</b>")
textbox110 = QLineEdit(w)
textbox110.move(500, 450)
textbox110.resize(150, 20)
textbox110.setText("Results")
button110 = QPushButton('', w)
button110.resize(20,20)
button110.move(670, 450)
button110.setStyleSheet("background-color: yellow");

label111 = QLabel(w)
label111.resize(200, 20)
label111.move(350, 500)
label111.setText("<b>RED GAP ALIGNMENT</b>")
textbox111 = QLineEdit(w)
textbox111.move(500, 500)
textbox111.resize(150, 20)
textbox111.setText("Results")
button111 = QPushButton('', w)
button111.resize(20,20)
button111.move(670, 500)
button111.setStyleSheet("background-color: yellow");

label9 = QLabel(w)
label9.move(800, 400)
label9.resize(200,20)
label9.setText("<b>TOTAL COUNT TODAY</b>")
textbox9 = QLineEdit(w)
textbox9.move(950, 400)
textbox9.resize(150, 20)
textbox9.setText(str(max_count))

##label10 = QLabel(w)
##label10.resize(200,20)
##label10.move(400, 600)
##label10.setText("<b>PART AT THE END OF THE BELT<b>")


decision_button = QPushButton(w)
decision_button.resize(300,300)
decision_button.move(800, 80)
##
##decision_button2 = QPushButton(w)
##decision_button2.resize(200,200)
##decision_button2.move(600, 500)


w.show()
app.exec_()
