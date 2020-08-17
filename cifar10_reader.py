# 加载相关数据处理的库
import numpy as np
import os
import random
import _pickle as cPickle


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
    
# 声明数据集文件位置
dir_data_batch1 = 'cifar-10-batches-py/data_batch_1'
dir_data_batch2 = 'cifar-10-batches-py/data_batch_2'
dir_data_batch3 = 'cifar-10-batches-py/data_batch_3'
dir_data_batch4 = 'cifar-10-batches-py/data_batch_4'
dir_data_batch5 = 'cifar-10-batches-py/data_batch_5'
dir_test_batch = 'cifar-10-batches-py/test_batch'

# unpickle训练集、测试集，生成数据集字典
train_set1_dict = unpickle(dir_data_batch1)
train_set2_dict = unpickle(dir_data_batch2)
train_set3_dict = unpickle(dir_data_batch3)
train_set4_dict = unpickle(dir_data_batch4)
train_set5_dict = unpickle(dir_data_batch5)
test_set_dict = unpickle(dir_test_batch)

# 基于unpickle的字典生成对应的训练集、测试集数据
train_set1 = list(train_set1_dict.values())
train_set2 = list(train_set2_dict.values())
train_set3 = list(train_set3_dict.values())
train_set4 = list(train_set4_dict.values())
train_set5 = list(train_set5_dict.values())
test_set = list(test_set_dict.values())

# 基于训练集、测试集数据得到数据集的图片数据切片
data_1 = train_set1[2]
data_2 = train_set2[2]
data_3 = train_set3[2]
data_4 = train_set4[2]
data_5 = train_set5[2]
data_test = test_set[2]

# 打印数据信息
print("训练数据集data_batch_1数量: ", len(data_1))
print("训练数据集data_batch_2数量: ", len(data_2))
print("训练数据集data_batch_3数量: ", len(data_3))
print("训练数据集data_batch_4数量: ", len(data_4))
print("训练数据集data_batch_5数量: ", len(data_5))
print("5个训练数据集总量: ", len(data_1)+len(data_2)+len(data_3)+len(data_4)+len(data_5))

print("测试数据集数量: ", len(data_test))



# 数据集相关参数，色彩通道数，图片高度IMG_ROWS, 图片宽度IMG_COLS
PIC_CHANNEL = 3
IMG_ROWS = 32
IMG_COLS = 32

# b'batch_label', b'labels', b'data', b'filenames'
batch_label_1, labels_1, data_1, filenames_1 = train_set1[0], train_set1[1], train_set1[2], train_set1[3]
batch_label_2, labels_2, data_2, filenames_2 = train_set1[0], train_set1[1], train_set1[2], train_set1[3]
batch_label_3, labels_3, data_3, filenames_3 = train_set1[0], train_set1[1], train_set1[2], train_set1[3]
batch_label_4, labels_4, data_4, filenames_4 = train_set1[0], train_set1[1], train_set1[2], train_set1[3]
batch_label_5, labels_5, data_5, filenames_5 = train_set1[0], train_set1[1], train_set1[2], train_set1[3]
batch_label_test, labels_test, data_test, filenames_test = test_set[0], test_set[1], test_set[2], test_set[3]

# 因为cifar的训练集和测试集的长度一致，此处就直接定义其长度为data_batch_1的长度
data_length = len(data_1)

# 定义数据集每个数据的序号，根据序号读取数据
index_list = list(range(data_length))

# 读入数据时用到的批次大小
BATCHSIZE = 100

# 随机打乱训练数据的索引序号
random.shuffle(index_list)

# 定义数据生成器，返回批次数据
def data_generator():
    data_list = []
    labels_list = []
    for i in index_list:
        # 将数据处理成期望的格式，比如类型为float32，shape为[3, 32, 32]
        img = np.reshape(data_1[i], [PIC_CHANNEL, IMG_ROWS, IMG_COLS]).astype('float32')
        label = np.reshape(labels_1[i], [1]).astype('float32')
        data_list.append(img)
        labels_list.append(label)
        if len(data_list) == BATCHSIZE:
            # 获得一个batchsize的数据，并返回
            yield np.array(data_list), np.array(labels_list)
            # 清空数据读取列表
            data_list = []
            labels_list = []

    # 如果剩余数据的数目小于BATCHSIZE，
    # 则剩余数据一起构成一个大小为len(data_list)的mini-batch
    if len(data_list) > 0:
        yield np.array(data_list), np.array(labels_list)
    return data_generator
    
    
# 声明数据读取函数，从训练集中读取数据
train_loader = data_generator

# 以迭代的形式读取数据
for batch_id, data in enumerate(train_loader()):
    image_data, label_data = data
    if batch_id == 0:
        # 打印数据shape和类型
        print("打印第一个batch数据的维度:")
        print("图像维度: {}, 标签维度: {}".format(image_data.shape, label_data.shape))
    break
    
    
# 机器校验：如果数据集中的图片数量和标签数量不等，说明数据逻辑存在问题，可使用assert语句校验图像数量和标签数据是否一致。
data_length = len(data_1)
assert len(data_1) == len(labels_1), "length of train_imgs({}) should be the same as train_labels({})".format(len(data_1), len(labels_1))


# 人工校验：打印数据输出结果，观察是否是预期的格式。实现数据处理和加载函数后，我们可以调用它读取一次数据，观察数据的shape和类型是否与函数中设置的一致
# 声明数据读取函数，从训练集中读取数据
train_loader = data_generator
# 以迭代的形式读取数据
for batch_id, data in enumerate(train_loader()):
    image_data, label_data = data
    if batch_id == 0:
        # 打印数据shape和类型
        print("打印第一个batch数据的维度，以及数据的类型:")
        print("图像维度: {}, 标签维度: {}, 图像数据类型: {}, 标签数据类型: {}".format(image_data.shape, label_data.shape, type(image_data), type(label_data)))
    break
