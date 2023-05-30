import os
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

__all__ = ['get_imglists','spilt_train_test']

IMAGE_FORMAT = ('.png', '.jpg', '.bmp')


def get_imglists(root, mode='train', spilt='train'):
    '''
    get all images path
    @param: 
        root : root path to dataset
        mode : method to read images
        spilt: sub path to specific dataset folder
    '''
    if mode == 'test':
        files = []
        # for imgname in tqdm(os.listdir(root)):
        #     files.append(imgname)
        files = list( map(lambda x: os.path.join(root, x), os.listdir(root)))
        files = pd.DataFrame({"filename": files})
        return files
    
    elif mode == 'inference':
        imgs, labels = [], []
        # current_folder = root  # 用于验证模型
        # folders = list(map(lambda x: os.path.join(current_folder, x),
        #                    os.listdir(current_folder)) ) # train0/1, test0/1, folder
        
        # for folder in folders:            
        #     print('loading {} dataset...'.format(folder.split(os.sep)[-2:]))
        #     imgslist = list(
        #         map(lambda x: os.path.join(folder, x), os.listdir(folder)))
        #     for filename in tqdm(imgslist):
        #         if filename.endswith(IMAGE_FORMAT):
        #             imgs.append(filename)
        #             labels.append(int(filename.split(os.sep)[-2]))
        # tt/20201210dataRJ/20201208RJ/rotate
        current_folder = os.path.join(root, 'test')  # train,val, or test   20201201find_best  20201204valid_mm
        imgs, labels = getFiles(current_folder)

        files = pd.DataFrame({'filename': imgs, 'label': labels})
        return files
    
    else:
        imgs, labels = [], []
        current_folder = os.path.join(root, spilt)  # train,val, or test
        # folders = list(map(lambda x: os.path.join(current_folder, x),
                        #    os.listdir(current_folder)) ) # train0/1, test0/1, folder
        imgs, labels = getFiles(current_folder)
        # for folder in folders:    
        #     print('loading {} dataset...'.format(folder.split(os.sep)[-2:]))
            # imgslist = list(
            #     map(lambda x: os.path.join(folder, x), os.listdir(folder)))
            
        # for filename in tqdm(imgslist):
        #     if filename.endswith(IMAGE_FORMAT):
        #         # imgs.append(filename)
        #         labels.append(int(filename.split(os.sep)[-2]))
        files = pd.DataFrame({'filename': imgs, 'label': labels})
        return files


def getFiles(dir, suffix='.png'): # 查找根目录，文件后缀 
    res = []
    labels = []
    dic = {}
    for root, directory, files in os.walk(dir):  # =>当前根,根下目录,目录下的文件
        for filename in files:
            name, suf = os.path.splitext(filename) # =>文件名,文件后缀
            if suf == suffix:
                if len(root.split('/')[-1])>2:
                    label = int(root.split('/')[-1][1])
                else:
                    label = int(root.split('/')[-1])
                # if label==7:
                #     label = 6
                res.append(os.path.join(root, filename)) # =>吧一串字符串组合成路径
                labels.append(label)
                
                if label in dic.keys():
                    dic[label]+=1
                else:
                    dic[label]=1

    print('---'*10,dic)
    return res, labels


def spilt_train_test(root, mode='train',spilt='train', test_ratio = 0.1):
    imgslist = get_imglists(root,mode,spilt)
    train_imgs, val_imgs = train_test_split(imgslist,test_size = test_ratio, stratify=imgslist["label"])
    return train_imgs, val_imgs


if __name__ == "__main__":
    root = '/home/fdd/dataset/v3_hunhe_0427'
    get_imglists(root, mode='train', spilt='train')
    
    get_imglists(root, mode='train', spilt='test')
    
