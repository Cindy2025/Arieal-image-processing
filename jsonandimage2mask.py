import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import os
import json


def PictureToMask(d_object, sourcePicture):
    '''得到原图的宽度和高度'''
    im = Image.open(sourcePicture)
    size = list(im.size)
    width = size[0]
    height = size[1]

    '''将图片的像素的宽度和高度换算成英寸的宽度和高度'''
    dpi = 100  # 分辨率
    ycwidth = width / dpi  # 宽度(英寸) = 像素宽度 / 分辨率
    ycheight = height / dpi  # 高度(英寸) = 像素高度 / 分辨率

    color = ["g", "r", "b", "y", "skyblue", "k", "m", "c"]
    fig, ax = plt.subplots(figsize=(ycwidth, ycheight))
    for region in d_object:
        '''将传进来的x轴坐标点和y轴坐标点转换成numpy数组，相加后转置成多行两列'''
        x = np.array(d_object[region][0])
        y = np.array(d_object[region][1]) * -1
        xy = np.vstack([x, y]).T
        # print(xy.shape)
        '''
        #设置画框的背景图片为原图
        fig = plt.figure(figsize=(ycwidth,ycheight),dpi=dpi)
        bgimg = img.imread(sourcePicture)
        fig.figimage(bgimg)
        '''
        '''将numpy中的坐标连城线，绘制在plt上'''
        # plt.plot(xy[:,0],xy[:,1],color=color[int(region)])
        # plt.fill_between(xy[:,0],xy[:,1],facecolor=color[int(region)])  #对该分割区域填充颜色
        plt.plot(xy[:, 0], xy[:, 1])
        plt.fill_between(xy[:, 0], xy[:, 1])  # 对该分割区域填充颜色
    plt.xticks([0, width])
    plt.yticks([0, -height])
    # plt.axis([0,0,1,1])
    # plt.axis("off")
    # 保存图片
    path = sourcePicture.rsplit(".", 1)[0]
    # print(sourcePicture)
    # print(path)
    # plt.savefig(path + "-mask.png", format='png', bbox_inches='tight', transparent=False, dpi=100) # bbox_inches='tight' 图片边界空白紧致, 背景透明
    # plt.savefig(path + "-mask.png", format='png', transparent=True, dpi=100)
    # plt.show()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.savefig(path + "-mask.png", format='png', transparent=True, dpi=100, pad_inches=0)


def getJson(filepath):
    '''从文件夹获取json文件内容，返回字典'''
    files = os.listdir(filepath)
    for file in files:
        if file.split(".")[1] == "json":
            jsonfile = filepath + file
            break
    jsonstr = open(jsonfile, "r", encoding="utf8").read()
    d_json = json.loads(jsonstr)
    # print(d_json)
    return d_json


def getPath():
    '''输入图片文件夹路径'''
    filepath = input("图片文件夹路径：")
    if filepath.endswith != "/" or filepath.endswith != "\\":
        filepath = filepath + "/"
    return filepath


def main():
    filepath = getPath()
    d_json = getJson(filepath)
    # print(d_json)
    d_json = d_json['_via_img_metadata']
    for key in d_json:
        # print(key)
        data = d_json.get(key)
        # print(data)
        pictureName = data["filename"]
        d_object = {}
        count = 0
        for region in data["regions"]:
            l_object = []
            # x = data["regions"][region]["shape_attributes"]["all_points_x"]
            # y = data["regions"][region]["shape_attributes"]["all_points_y"]
            x = region["shape_attributes"]["all_points_x"]
            y = region["shape_attributes"]["all_points_y"]
            x.append(x[0])
            y.append(y[0])
            l_object.append(x)
            l_object.append(y)
            d_object[str(count)] = l_object
            count += 1
        sourcePicture = filepath + pictureName
        PictureToMask(d_object, sourcePicture)


if __name__ == "__main__":
    main()