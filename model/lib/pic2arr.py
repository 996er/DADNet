import io

import numpy as np

import os


from PIL import Image
import struct


path1='/home/yu/公共的/ganomaly/data/testMain/fakes.png'

path2='/home/yu/公共的/ganomaly/data/testMain/reals.png'

path3='/home/yu/公共的/ganomaly/output/ganomaly/testMain/train/images/fixed_fakes_004.png'

path4='/home/yu/公共的/ganomaly/data/testMain/train/1227860816/id:000000,src:000000,op:flip1,pos:0,1227860816.jpg'

path5='/home/yu/testC/output/train/id:000000,src:000000,op:flip1,pos:0,1227860816'

path6='/home/yu/桌面/testpic/1.txt'

testArray = np.zeros((4, 4), dtype='uint8')

print(np.fromfile(path5))

with open(path5,'rb') as f:
    data = f.read()
    lenth = len(data)
    data=np.frombuffer(data,dtype='uint8')
    print(data)
    if lenth > 16:
        lenth = 16
    # print(lenth)
    for i in range(lenth):
        # print(data[i])
        line = i // 4
        row = i % 4
        testArray[line][row] = data[i]

    print(testArray)

    conv2pic = Image.fromarray(np.uint8(testArray), "L")
    print(conv2pic.getdata()[14])



    conv2pic.save('/home/yu/桌面/testpic/1.jpg',quality=100)
    # misc.imsave('/home/yu/桌面/testpic/1.jpg', conv2pic)

    # conv2pic=np.matrix(conv2pic)
    #
    # print(conv2pic)



im = Image.open('/home/yu/桌面/testpic/1.jpg')


# mat= np.asarray(im)
#
# print(mat)

picArr=np.matrix(im)
# print(picArr.shape)
print(picArr)



# with open(path6, 'wb')as fp:
#     for x in list_dec:
#         a = struct.pack('B', x)
#         fp.write(a)





