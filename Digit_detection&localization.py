import numpy as np
import cv2                                                                    #import opencv

#Teams can add other helper functions
#which can be added here

cellnos1, cellnos2, digits_grid1, digits_grid2 = [[] for i in range(4)]       #create empty lists to store digits & cell numbers in division1 & division2

def digit_detection(x,y,w,h,thresh):                                          #function to detect digits using K-Nearest Neighbor
    samples = np.loadtxt('trainingsamples.data',np.float32)                   #Loading the data that was generated after training
    responses = np.loadtxt('trainingresponses.data',np.float32)       
    responses = responses.reshape((responses.size,1))
    model = cv2.KNearest()                                                    #training K-Nearest Neighbor model
    model.train(samples,responses)
    roi = thresh[y:y+h,x:x+w]                                                 #selecting the sample in the bounding rectangle
    roismall = cv2.resize(roi,(10,10))                                        #resizing the box to 10x10 
    roismall = roismall.reshape((1,100))                             
    roismall = np.float32(roismall)                                           #saves 100 pixel values in an array
    retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)    #finds nearest neighbour
    return (int((results[0][0])))                                               #return digit

def cellnumber(p,q):                                                            #function to find cell number of digit
    if 58 < p <398:                                                             #check for division D1 and assign values of row and columns
        k = 3
        l = 4
        xmin = 58                                                               #minimum value of x in Division 1                                                               
    else :                                                                      #check for division D2 and assign values of row and columns
        k = 4
        l = 6
        xmin = 525                                                              #minimum value of x in Division 2
    for i in range(k):                                                          #logic to find cell number
        for j in range(l):                                                            
            a = 85*j                                                            #x coordinate of staring point of a cell                                         
            b = 85*i                                                            #y coordinate of staring point of a cell  
            c = (p - xmin)                                                      #c is the modified value of x coordinate of bounding rectangle
            if ((85*j)+5 < c < a+44) & (b+28 < q < b+40):                       #position of top left coordinates(x,y) of bounding rectangle is checked
                if a+5 < c < a+30:                                              #if it lies before the refrence point(starting point of a cell+30),
                    m = (6*i)+j                                                 #then cell number is calulated and store
                else:                                                           #if it lies after the refrence point(i.e.the 2nd digit in 2digit no)
                    m = (6*i)+j+0.5                                             #then float number(0.5) is added to the cell no(to indicate it as a 2nd digit)
                if 58 < p <398:                                                 #if the contour is in 1st grid,append it to list 'cellnos1'
                    cellnos1.append(m)
                else :                                                          #else append it to list 'cellnos2'
                    cellnos2.append(m)


def sortlists(list1,list2):                                                     #function to sort lists
                                                                                #sorting digits by sorting cellnumbers in ascending order
    together = zip(list1,list2)                                                 #zip the lists together 
    sorted_together = sorted(together)                                          #sort them
    l1 = [x[0] for x in sorted_together]                                        #unzip the list
    l2 = [x[1] for x in sorted_together]                                        #l1 contains cellnos and l2 contains digits after sorting
    sorted_list = zip(l1,l2)                                                    #l1 and l2 are zipped together
    list3 = [list(i) for i in sorted_list]                                      #tuple is converted into list to modify the elements of tuples
    for i in range(len(list3)):
        if isinstance((list3[i][0]), float):                                    #checks if a cell number is a float?
            list3[i-1][1] = int((str(list3[i-1][1]))+(str(list3[i][1])))        #if yes then concatenate it with previous digit(as it is 2nd digit of 2digit no)
            list3[i][0]=100
            list3[i][1]=100                                                     #replace the contents by [100,100](as max digit in grid2 is 99)
    list4 = [l for l in list3 if l[0] != 100 and l[1] != 100]                   #create a new list from elements of list3 by excluding element[100,100] in a list3
    return l2,list4                                                             #return l2 and list4
    
    
def play(img):
    '''
    img-- a single test image as input argument
    No_pos_D1 -- List containing detected numbers in Division D1
    No_pos_D2 -- List of pair [grid number, value]

    '''
    
    #add your code here
    roii = img[72:435,:,:]                                                      #taking a region of interest from image
    gray = cv2.cvtColor(roii,cv2.COLOR_BGR2GRAY)                                #convert an image to gray
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)                           #Applies an adaptive threshold

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  #find contours
    for count,cnt in enumerate(contours):                                       #for cnt in contours    
        if 30 < cv2.arcLength(cnt,True) < 250:                                  #condition for the contours to be a digit
            [x,y,w,h] = cv2.boundingRect(cnt)                                   #make a bounding rectangle around the contour
            if  (h < 60) & (w < 60):                                            #condition to avoid grids
                cv2.drawContours(roii,[cnt],0,(0,255,0),2)                      #draw contours
                if 100 < cv2.arcLength(cnt,True) < 250:                         #condition to neglect inner contour of digits like 0,9,6,8,4
                    if 58 < x < 398:                                            #divison1
                        
                        digits_grid1.append(digit_detection(x,y,w,h,thresh))    #call a function 'digit_detection' & append the returned value to 'digits_grid1'
                        cellnumber(x,y)                                         #call function 'cellnumber'
                        
                    elif 525 < x < 1035:                                        #division2
                        
                        digits_grid2.append(digit_detection(x,y,w,h,thresh))    #call a function 'digit_detection' & append the returned value to 'digits_grid2'
                        cellnumber(x,y)                                         #call function 'cellnumber'

    No_pos_D1, L1 = sortlists(cellnos1,digits_grid1)                            #call function 'sortlists' to sort lists 'cellnos1' & 'digit_grid1'
    L2, No_pos_D2 = sortlists(cellnos2,digits_grid2)                            #call function 'sortlists' to sort lists 'cellnos2' & 'digit_grid2'
    
    return No_pos_D1, No_pos_D2                                                 #return 'No_pos_D1' and 'No_pos_D2'


if __name__ == "__main__":                                                      #Main function
    #code for checking output for single image
    img = cv2.imread('test_images/test_image1.jpg')                             #read the image
    No_pos_D1,No_pos_D2 = play(img)                                             #call 'play' function
    print 'D1 =',No_pos_D1                                                      #print list 'No_pos_D1'
    print 'D2 =',No_pos_D2                                                      #print list 'No_pos_D2'
    cv2.imshow('im',img)                                                        #show the image
    cv2.waitKey(0)                                                              #wait for the user to press key
    cv2.destroyAllWindows()                                                     #close windows
