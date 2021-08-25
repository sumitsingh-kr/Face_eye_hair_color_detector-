import pandas as pd
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN 

detector = MTCNN()
cap = cv2.VideoCapture(0)

csv_path_face = 'color5.xlsx'
csv_path_hair = 'color6.xlsx'
index = ['color', 'color_name', 'hex', 'R', 'G', 'B','R1', 'G1', 'B1']
df_f = pd.read_excel(csv_path_face, names=index, header=None)
df_h = pd.read_excel(csv_path_hair, names=index, header=None)

# define HSV color ranges for eyes colors
class_name = ("Blue", "Blue Gray", "Brown", "Brown Gray", "Brown Black", "Green", "Green Gray", "Other")
EyeColor = {
    class_name[0] : ((166, 21, 50), (240, 100, 85)),
    class_name[1] : ((166, 2, 25), (300, 20, 75)),
    class_name[2] : ((2, 20, 20), (40, 100, 60)),
    class_name[3] : ((20, 3, 30), (65, 60, 60)),
    class_name[4] : ((0, 10, 5), (40, 40, 25)),
    class_name[5] : ((60, 21, 50), (165, 100, 85)),
    class_name[6] : ((60, 2, 25), (165, 20, 65))
}
#_______________________________________________________________________________________________________________________
def check_color(hsv, color):
    if (hsv[0] >= color[0][0]) and (hsv[0] <= color[1][0]) and (hsv[1] >= color[0][1]) and \
    hsv[1] <= color[1][1] and (hsv[2] >= color[0][2]) and (hsv[2] <= color[1][2]):
        return True
    else:
        return False
#_______________________________________________________________________________________________________________________
# define eye color category rules in HSV space
def find_class(hsv):
    color_id = 7
    for i in range(len(class_name)-1):
        if check_color(hsv, EyeColor[class_name[i]]) == True:
            color_id = i

    return color_id
#_____________________________________________________________________________________________________________________
#define RGB for face
def get_color_name_face(R,G,B):
    minimum = 1000
    for i in range(len(df_f)):
        Rd = ((df_f.loc[i,'R'])+(df_f.loc[i,'R1']))/2
        Gd = ((df_f.loc[i,'G'])+(df_f.loc[i,'G1']))/2
        Bd = ((df_f.loc[i,'B'])+(df_f.loc[i,'B1']))/2
        d = abs(R - int(Rd)) + abs(G - int(Gd)) + abs(B - int(Bd))
        if d <= minimum:
            minimum = d
            cname = df_f.loc[i, 'color_name']

    return cname
#______________________________________________________________________________________________________________________
#define RGB for hair
def get_color_name_hair(R,G,B):
    minimum = 1000
    for i in range(len(df_h)):
        Rd = ((df_h.loc[i,'R'])+(df_h.loc[i,'R1']))/2
        Gd = ((df_h.loc[i,'G'])+(df_h.loc[i,'G1']))/2
        Bd = ((df_h.loc[i,'B'])+(df_h.loc[i,'B1']))/2
        d = abs(R - int(Rd)) + abs(G - int(Gd)) + abs(B - int(Bd))
        if d <= minimum:
            minimum = d
            cname = df_h.loc[i, 'color_name']

    return cname
#_____________________________________________________________________________________________________________________
def eye_color(image):
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[0:2]
    imgMask = np.zeros((image.shape[0], image.shape[1], 1))
    result = detector.detect_faces(image)
    if result == []:
        print('Warning: Can not detect any face in the input image!')
        return

    left_eye = result[0]['keypoints']['left_eye']
    right_eye = result[0]['keypoints']['right_eye']

    eye_distance = np.linalg.norm(np.array(left_eye)-np.array(right_eye))
    eye_radius = eye_distance/15 # approximate

    cv2.circle(image, left_eye, int(eye_radius), (0, 155, 255), 1)
    cv2.circle(image, right_eye, int(eye_radius), (0, 155, 255), 1)

    eye_class = np.zeros(len(class_name), np.float)

    #Key range
    for y in range(0, h):
        for x in range(0, w):
            if imgMask[y, x] != 0:
                eye_class[find_class(imgHSV[y,x])] +=1 

    main_color_index = np.argmax(eye_class[:len(eye_class)-1])
    total_vote = eye_class.sum()
    
    print("\n\nDominant Eye Color: ", class_name[main_color_index])
    # print("\n **Eyes Color Percentage **")

    # for i in range(len(class_name)):
        # print(class_name[i], ": ", round(eye_class[i]/total_vote*100, 2), "%")
    
    label = 'Eye Color: %s' % class_name[main_color_index]

    cv2.putText(image, label, (left_eye[0]-10, left_eye[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155,255,0))

    cv2.imshow('COLOR-DETECTION', image)
#____________________________________________________________________________________________________________________________________
# Define Hair Color    
def hair_color(image):
    result = detector.detect_faces(image)
    bounding_box = result[0]['box'] 
    centerCoord_hair = (bounding_box[0]+(bounding_box[2]/2), (bounding_box[1]-(bounding_box[3]/2)-35)+(bounding_box[3]/2))
    cv2.circle(image,(int(centerCoord_hair[0]),int(centerCoord_hair[1])),5,(0,155,255), 2)
    y = int(centerCoord_hair[0])
    x = int(centerCoord_hair[1])
    (b, g, r) = image[y,x]
    print("Hair color: ", get_color_name_hair(b,g,r)) 

    label =  'Hair Color: %s' % get_color_name_hair(b,g,r)
    cv2.putText(image,label,(bounding_box[0],bounding_box[1]-int(bounding_box[3]/3)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155,255,0))
    cv2.imshow('COLOR-DETECTION', image)
#_________________________________________________________________________________________________________________________________________
# Define Face Color    
def face_color(image):
    result = detector.detect_faces(image)
    bounding_box = result[0]['box'] 
    forehead = ((bounding_box[0]+(bounding_box[2]/2)),(bounding_box[1]+(bounding_box[3]/6)))
    cv2.circle(image,(int(forehead[0]),int(forehead[1])),5,(0,155,255), 2)
    y = int(forehead[0])
    x = int(forehead[1])
    (b, g, r) = image[y,x]
    print("Face COlor: ",get_color_name_face(b,g,r))
    label =  'Face Color: %s' % get_color_name_face(b,g,r)
    p = (bounding_box[0]+(bounding_box[2]/6))
    q = (bounding_box[1]+(bounding_box[3]/2))
    cv2.putText(image,label,(int(p),int(q)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155,255,0))
    
    cv2.imshow('COLOR-DETECTION', image)
#_____________________________________________________________________________________________________________________________________________

if __name__ == '__main__':

        while(1):
            ret, image = cap.read()
            if ret == -1: 
                break

            eye_color(image)
            hair_color(image)
            face_color(image)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break
        #When everything's done, release capture
        cap.release()
        cv2.destroyAllWindows()