import requests
import os
from lxml import html
import numpy as np
import cv2

def download_data(url, params):
    page = requests.get(url, params=params)
    tree = html.fromstring(page.content)
    data_links = tree.xpath('//a/@href')
    
    path = 'UCF101' + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    print(os.listdir(path))
    
    for i in range(5, len(data_links)):
        filename = data_links[i]
        video = requests.get(url + filename, params=params)
        open(path + filename, 'wb').write(video.content)
        
        if i % 50 == 0:
            print(i - 4, "files written")

    print(len(data_links) - 5, "files written")

def get_frames(path):
    
    cap = cv2.VideoCapture(path)
    frames = []
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(gray, (120, 90), interpolation = cv2.INTER_AREA)
            frames.append(frame)
        else:
            break

    cap.release()
    return np.array(frames)
    
def trim_len(total_frames, no_frames=20):
    
    data_size = total_frames // no_frames
    data_size = no_frames * data_size
    return data_size

def create_dataset(path):
    
    datasets = None
    cur_action = None
    indices = [(0, '')]
    
    files = os.listdir(path)
    files.sort()
    
    for filename in files:
        sample = get_frames(path + filename)
        
        if datasets is None:
            datasets = sample
            cur_action = filename
        else:
            datasets = np.concatenate((datasets, sample), axis=0)
            if filename[:-12] != cur_action[:-12]:
                datasets = datasets[:trim_len(datasets.shape[0])]
                indices.append((datasets.shape[0] - trim_len(sample.shape[0]), cur_action[:-12]))
                cur_action = filename
            elif filename[:-8] != cur_action[:-8]:
                datasets = datasets[:trim_len(datasets.shape[0])]
                cur_action = filename
                
        
    datasets = datasets[:trim_len(datasets.shape[0])]
    indices.append((datasets.shape[0], cur_action[:-12]))
    indices = np.array(indices)
    
    return datasets, indices
    
def main():
    
    url = 'http://crcv.ucf.edu/THUMOS14/UCF101/UCF101/'
    params = {}
    download_data(url, params)
    
    datasets, indices = create_dataset('./UCF101/')
    
    print(datasets.shape, indices.shape)
    
    np.save('./UCF101/ucf101.npy', datasets)
    np.save('./UCF101/ucf_idx.npy', indices)

if __name__ == "__main__":
    main()
