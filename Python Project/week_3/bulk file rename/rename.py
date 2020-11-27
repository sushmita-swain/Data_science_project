import os 
folderpath = r'D:\Python Project\bulk file rename\images'
fileNumber = 1

for filename in os.listdir(folderpath):
    os.rename(folderpath + '\\' + filename, folderpath + '\\' + "food_" + str(fileNumber)  + '.jpg')
    fileNumber +=1