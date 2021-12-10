import torch
from torch.utils.data import Dataset
import json
import os
from collections import Counter
from random import choice, sample
from PIL import Image


class CaptionDataset(Dataset):
    def __init__(self, data_folder="/home/disk_4T_0/zhangboyuan/dataRoot/MSCOCO14",
                 split='train', transform=None):
        """
        :param data_folder: folder where data files are stored, below are train/val and .json
        :param split: split, one of 'train', 'val', or 'test'
        :param transform: image transform pipeline
        """
        self.data_folder = data_folder
        self.split = split
        assert self.split in {'train', 'val', 'test'}
        self.transform = transform

        self.get_basic()
        # get self.image_paths self.image_captions, self.word_map w.r.t self.split
        self.augment_caption()
        # sample self.image_captions so that self.cpi = 5
        self.caption2num()
        # add <start>, <end> and <pad>, then convert words to number
        # get self.caplens self.enc_captions

        self.cpi = 5

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.enc_captions)

    def caption2num(self):
        # add <start>, <end> and <pad>, then convert words to number
        self.caplens = []
        enc_captions = []
        for image_caption in self.image_captions:
            # Encode captions
            for caption in image_caption:
                enc_c = [self.word_map['<start>']] + \
                        [self.word_map.get(word, self.word_map['<unk>']) for word in caption] + \
                        [self.word_map['<end>']] + \
                        [self.word_map['<pad>']] * (self.max_len - len(caption))

                # Find caption lengths
                c_len = len(caption) + 2
                # 52 usually
                enc_captions.append(enc_c)
                self.caplens.append(c_len)
        self.enc_captions = enc_captions
        # print(self.caplens[:10])
        # print(self.image_captions[:10])
        # print(self.enc_captions[:10])
        assert len(self.enc_captions) == len(self.caplens)

    def augment_caption(self):
        """
        :return: sample self.image_captions so that self.cpi = 5
        """
        # augment self.image_captions to cpi == 5
        new = []
        for image_caption in self.image_captions:
            if len(image_caption) < self.cpi:
                captions = image_caption + [choice(image_caption) for _ in range(self.cpi - len(image_caption))]
            else:
                captions = sample(image_caption, k=self.cpi)
            new.append(captions)
        self.image_captions = new
        total_caption = sum([len(image_caption) for image_caption in self.image_captions])
        assert len(self.image_paths) * 5 == total_caption

    def get_basic(self):
        """
        :return: self.image_paths self.image_captions, self.word_map
        """
        json_path = os.path.join(self.data_folder, "dataset_coco.json")
        self.max_len = 50
        self.min_word_freq = 5
        self.cpi = 5

        with open(json_path, 'r') as j:
            data = json.load(j)

        train_image_paths = []
        train_image_captions = []
        val_image_paths = []
        val_image_captions = []
        test_image_paths = []
        test_image_captions = []
        word_freq = Counter()

        # print(data['images'][0])
        # print(data['images'][0]['sentences'])
        # 一个图片的句子描述是一个列表，里面是多个字典，每个字典
        # 里面有tokens (单词的列表), raw (连续的句子), imgid, sentid

        image_folder = self.data_folder
        # 该folder下有train2014/val2014

        for img in data['images']:
            # 总共有123287个image
            captions = []
            for c in img['sentences']:
                word_freq.update(c['tokens'])
                if len(c['tokens']) <= self.max_len:
                    captions.append(c['tokens'])

            if len(captions) == 0:
                continue

            path = os.path.join(image_folder, img['filepath'], img['filename'])
            # filepath为"train2014"或者"val2014"
            # filename为图像的名称

            # img["split"]为val, test, restval, train中的一个

            if img['split'] in {'train', 'restval'}:
                train_image_paths.append(path)
                train_image_captions.append(captions)
            elif img['split'] in {'val'}:
                val_image_paths.append(path)
                val_image_captions.append(captions)
            elif img['split'] in {'test'}:
                test_image_paths.append(path)
                test_image_captions.append(captions)

        # print(len(word_freq.keys()))
        # 27929
        if self.split == 'train':
            self.image_paths = train_image_paths
            self.image_captions = train_image_captions
        elif self.split == 'val':
            self.image_paths = val_image_paths
            self.image_captions = val_image_captions
        else:
            self.image_paths = test_image_paths
            self.image_captions = test_image_captions

        words = [w for w in word_freq.keys() if word_freq[w] > self.min_word_freq]
        word_map = {k: v + 1 for v, k in enumerate(words)}
        # {"a": 1, "the": 2, ...}
        word_map['<unk>'] = len(word_map) + 1
        word_map['<start>'] = len(word_map) + 1
        word_map['<end>'] = len(word_map) + 1
        word_map['<pad>'] = 0
        self.word_map = word_map

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        image_path = self.image_paths[i // self.cpi]
        image = Image.open(image_path).convert("RGB")
        # add convert RGB so that channel dimension is always 3

        if self.transform is not None:
            image = self.transform(image)

        word_caption = self.image_captions[i // self.cpi]
        # return a list with five captions
        number_caption = torch.LongTensor(self.enc_captions[i])
        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'train':
            # return image, number_caption, caplen, word_caption
            return image, number_caption, caplen
        else:
            # For validation or testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.enc_captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return image, number_caption, caplen, all_captions
            # return image, number_caption, caplen, all_captions, word_caption

    def __len__(self):
        return self.dataset_size


if __name__ == "__main__":
    dataset = CaptionDataset(data_folder="F://NLP大作业数据集//MSCOCO14", split='test')
    # train 113287 val 5000 test 5000 (\* 5 respectively
    a = dataset.__getitem__(333)
    print(a[-1].shape)
    print(a[-1])
    # image = a[0]
    # image.show()
    # print(a)

    # import json
    # with open('./word_map.json', 'w') as f:
    #     json.dump(dataset.word_map, f)
