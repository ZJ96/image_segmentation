import numpy as np

def get_batch_img(img,overlap_pix = 30):
    w = img.shape[0]
    h = img.shape[1]
    if w ==256 and h ==256:
        return np.array([img])
    a, b = 256, 256
    res = []
    gap = 256 - overlap_pix

    while a < w:
        while b < h:
            res.append(img[a - 256:a, b - 256:b, :])
            b += gap
        res.append(img[a - 256:a, h - 256:h, :])
        a += gap
        b = 256
    while b < h:
        res.append(img[w - 256:w, b - 256:b, :])
        b += gap
    res.append(img[w - 256:w, h - 256:h, :])
    return np.array(res)

def merge_label(label,w,h,overlap_pix = 30):
    if w==256 and h==256:
        return label[0]

    pix = int(overlap_pix / 2)
    gap = 256 - overlap_pix

    #the first col
    a, b = 256, 256
    res = label[0][0:256-pix,0:256-pix,:]
    i = 1
    while b + gap < h:
        res = np.hstack([res, label[i][0:256-pix,pix:256-pix,:]])
        i += 1
        b += gap
    if b!=h:
        res = np.hstack([res, label[i][0:256-pix,256 - (h-b+pix):256,:]])
        i += 1

    #center col
    while a + gap <w:
        tmp = label[i][pix:256-pix,0:256-pix,:]
        i += 1
        b = 256
        while b + gap < h:
            tmp = np.hstack([tmp, label[i][pix:256-pix,pix:256-pix,:]])
            i += 1
            b += gap
        if b!=h:
            tmp = np.hstack([tmp, label[i][pix:256-pix, 256 - (h-b+pix):256 ,:]])
            i += 1
        res = np.vstack([res, tmp])
        a += gap

    #the last col
    if a!=w:
        tmp = label[i][256 - (w-a+pix):256, 0:256-pix,:]
        i += 1
        b = 256
        while b + gap < h:
            tmp = np.hstack([tmp, label[i][256 - (w-a+pix):256, pix:256-pix,:]])
            i += 1
            b += gap
        if b!=h:
            tmp = np.hstack([tmp, label[i][256 - (w-a+pix):256, 256 - (h-b+pix):256,:]])
            i += 1
        res = np.vstack([res, tmp])

    return res

def test(img,w,h):
    # import cv2
    # img = cv2.imread("./001.jpg",cv2.IMREAD_UNCHANGED)
    # print(img.shape)
    # w = img.shape[0]
    # h = img.shape[1]
    img_split = get_batch_img(img)
    img_merge = merge_label(img_split,w,h)
    #print(img_merge.shape)
    if img.shape!=img_merge.shape:
        return False
    for i in range(w):
        for j in range(h):
            for k in range(3):
                if img[i][j][k]!=img_merge[i][j][k]:
                    return False
    return True

for i in range(4444,5000):
    img  = np.random.random((i,i,3))*100
    print(i)
    if test(img,i,i)==False:
        print("size {} is not True!".format(i))
        break