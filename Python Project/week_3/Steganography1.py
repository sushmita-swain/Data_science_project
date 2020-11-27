# reads immage, set key and enter text
import cv2

x = cv2.imread("images/food_1.jpg")

# print dimension of image
i = x.shape[0]
j = x.shape[1]

key = input("Enter key to edit : ")
text = input("Enter text to show : ")

# Hide data in image in pixels

for i in range(l):
    x[n, m, z] = d[text[i]] ^ d[key[kl]]
    n = n+1
    m = m+1
    # this is for every value of z , remainder will be between 0,1,2 . i.e G,R,B plane will be set automatically.
    m = (m+1) % 3
    # whatever be the value of z , z=(z+1)%3 will always between 0,1,2 . The same concept is used for random number in dice and card games.
    kl = (kl+1) % len(key)

