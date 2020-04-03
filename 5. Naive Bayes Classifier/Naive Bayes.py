from PIL import Image,ImageDraw
from sklearn.naive_bayes import GaussianNB
import numpy as np

im_x_train = Image.open('family.jpg')
im_y_train = Image.open('family.png')
im_x_test = Image.open('portrait.jpg')
im_y_real = Image.open('portrait.png')
w,h=im_x_train.size #get the size of the training picture
wTest,hTest = im_x_test.size# get the size of the test picture

x_train = []
y_train = []
x_test = []
y_real = []
for x in range(w):#get the RGB value in each pixel, and calculate r and g
    for y in range(h):
        rx, gx, bx = im_x_train.convert('RGB').getpixel((x,y))
        ry, gy, by = im_y_train.convert('RGB').getpixel((x,y))
        if rx+gx+bx!=0:
            r = rx/(rx+gx+bx)
            g = gx/(rx+gx+bx)
        else:#if rx and gx and bx are all 0, then equals to 1/3
            r = 1/3
            g = 1/3
        x_train.append(r)#put the calculated r and g as X in training samples
        x_train.append(g)
        y_train.append(ry)#the pixel in ground truth image are black or white, so just take the R value in RGB
x_train = np.array(x_train).reshape(-1,2)

for x in range(wTest):#get x_test, same as above
    for y in range(hTest):
        rxT,gxT,bxT = im_x_test.convert('RGB').getpixel((x,y))
        if rxT+gxT+bxT!=0:
            r = rxT/(rxT+gxT+bxT)
            g = gxT/(rxT+gxT+bxT)
        else:
            r = 1/3
            g = 1/3
        x_test.append(r)
        x_test.append(g)
        R,G,B = im_y_real.convert('RGB').getpixel((x,y))
        y_real.append(R)
x_test = np.array(x_test).reshape(-1,2)

clf=GaussianNB().fit(x_train,y_train) # train

y_test = clf.predict(x_test) # predict

#draw mask picture
image = Image.new('RGB', (wTest, hTest), (255, 255, 255))
draw = ImageDraw.Draw(image)
 
#decide the each pixels color in result picture
i=0
for x in range(wTest):
    for y in range(hTest):
        draw.point((x, y), fill=(y_test[i],y_test[i],y_test[i]))
        i=i+1
image.save('result.png', 'png')

skin=0 # number of pixels that are skin 
deteced_skin_in_skin=0 #number of pixels that are skin and predicted as skin
ground=0# number of pixels that are background(non-skin)
deteced_ground_in_ground=0 #number of pixels that are background and predicted as background
predict_skin_in_ground=0 #number of pixels that are background but predicted as skin
predict_ground_in_skin=0 #number of pixels that are skin but predicted as background

for i in range(wTest*hTest):
    if y_real[i]==0:
        ground = ground+1
        if y_test[i]==0:
            deteced_ground_in_ground = deteced_ground_in_ground+1
        else:
            predict_skin_in_ground=predict_skin_in_ground+1
    else:
        skin = skin+1
        if y_test[i]==255:
            deteced_skin_in_skin=deteced_skin_in_skin+1
        else:
            predict_ground_in_skin=predict_ground_in_skin+1

print("true positive rate =",(deteced_skin_in_skin/skin))
print("true negative rate =",(deteced_ground_in_ground/ground))
print("false positive rate =",(predict_skin_in_ground/ground))
print("false negative rate =",(predict_ground_in_skin/skin))