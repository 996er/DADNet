



import os

root="/home/yu/公共的/FoRTE-FuzzBench/cjson/out_dir_AD1/train/1"
filenames=os.listdir(root)
hash_value=-1170482070
names=[]

for name in filenames:
    fileName=name.split(',')[-1]  #hash值


    if fileName == str(hash_value):     #把hash值相同的先存下来
        names.append(name)

#以二进制的方式读取各个文件

diff=[]

lenth=len(names)

for i in range(0,lenth-1,2):
    with open(os.path.join(root, names[i]),"rb") as f:
        content1=f.read()
        print(content1)
        print("------------")
        f.close()
    with open(os.path.join(root, names[i+1]),"rb") as q:
        content2=q.read()
        print(content2)
        q.close()

    con1=bytearray(content1)  #0001 1101
    con2=bytearray(content2)  #0110 0010   0110 1100  0000 1110  0111 0001
                              #0111 1111  127 0111 1111
    length1=len(con1)
    length2=len(con2)

    if length1 < length2:

        for j in range(0,length1):
            print(con1[0])
            print("!")
            print(con2[0])
            diff=con1[0] ^ con2[0]
            print("---")
            print(diff)
    else:
        for j in range(0,length2):
            diff[j]=con1[j] ^ con2[j]
            print("---")
            print(diff[j])









#对相邻的文件做位异或处理，找出相同的位


