import numpy as np

import os

from PIL import Image


#image = Image.open('/home/yu/公共的/ganomaly/data/jpeg/train/1.normal/1.id:000025,-1167556048.jpeg')
#image.show()
#mat= np.array(image)
#print(mat)


def data2array():
    # path = '/home/yu/公共的/ganomaly/data/jpeg/train/1.normal/1.id:000025,-1167556048.jpeg'
    paths = '/home/yu/公共的/ganomaly/data/testMain/train'

    filelist = os.listdir(paths)   #train下面的所有文件

    testArray = np.zeros((4, 4), dtype='uint8')
    # print(testArray)
    for path in filelist:

        fileName=path.split(',')[-1]      #切出字符串hash值，文件名

        #print(fileName)

        pathName=os.path.join(paths, fileName)  #文件夹路径名，paths+fileName


        if not os.path.exists(pathName):
            os.mkdir(pathName)                #创建以hash值为名的文件

        #print(pathName)

        curPath=os.path.join(paths, path)

        #print(curPath)

        newPathName=os.path.join(pathName,path)  #新文件夹下的文件路径名

        #print(newPathName)



        with open(curPath, 'rb') as f:
            data = f.read()
            lenth = len(data)
            if lenth>16:
                lenth=16
            # print(lenth)
            for i in range(lenth):
                line = i // 4
                row = i % 4
                testArray[line][row] = data[i]



            # f.write(testArray)

            newpic=Image.fromarray(np.uint8(testArray),"L")
            # print(newpic)
            # picArr=np.matrix(newpic)
            # print(picArr)

            os.remove(curPath)

            newpic.save(newPathName+'.jpeg')



data2array()