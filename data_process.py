from datasets import Dataset
import codecs
import re
from datasets import load_from_disk
import glob
from datasets import concatenate_datasets


def split_train_test_long_text(long_text, long_tags):
    """
    中宣部全总北京市举办学习再就业先进典型座谈会北京多云转晴１５℃／２７℃天津多云１６℃／２８℃石家庄阴转多云１６℃／２７℃太原多云１３℃／２９℃呼和浩特晴１１℃／２６℃沈阳小雨转晴１３℃／２７℃大连多云转晴１６℃／２３℃长春多云转晴１４℃／２４℃哈尔滨多云转晴９℃／２５℃上海晴２０℃／３０℃南京雷阵雨１７℃／２９℃杭州多云转小雨２０℃／３１℃合肥多云１９℃／２９℃福州阴转多云２２℃／３０℃南昌多云转阴２０℃／２８℃济南多云２３℃／３０℃青岛多云转晴１６℃／２６℃郑州多云转阴１８℃／２７℃武汉晴转多云２０℃／３２℃长沙多云２０℃／３１℃广州多云转小雨２５℃／３２℃南宁小雨２５℃／３２℃海口多云２６℃／３５℃成都阴转雷阵雨２２℃／３０℃重庆多云转阴２２℃／３２℃贵阳小雨转多云１８℃／２９℃昆明多云转晴１７℃／２６℃拉萨多云６℃／２３℃西安小雨１６℃／２５℃兰州晴转多云１３℃／２９℃西宁多云１０℃／２２℃银川多云转阴１４℃／２３℃乌鲁木齐小雨转多云３℃／８℃台北阴２３℃／２９℃香港小雨２４℃／３０℃澳门小雨２４℃／３０℃东京多云１６℃／２４℃曼谷雷阵雨２７℃／３１℃悉尼晴１５℃／２２℃卡拉奇晴２６℃／３４℃开罗晴１９℃／３０℃莫斯科小雨９℃／１８℃法兰克福晴９℃／２４℃巴黎多云１４℃／２４℃伦敦多云转阴１３℃／２３℃纽约小雨１２℃／２０℃据中央气象台提供的信息：受冷空气和暖湿气流影响，１９日晚上到２０日，新疆北部、青藏高原东部、西北地区东南部、华北中南部、东北地区南部、黄淮北部、西南地区东部、华南大部将有阵雨或雷阵雨；新疆北部、甘肃西部、内蒙古西部以及青海北部有４—６级偏北或偏西风。
    """

    long_sub_sens, long_sub_tags = [], []
    last_postion = len(long_text)
    temp_end = 0
    for position in re.finditer(r'[\u4e00-\u9fa5]+[^\s]+?℃／[^\s]+?℃', long_text):
        sub_start = position.start()
        sub_end = position.end()
        sub_sen = long_text[sub_start: sub_end]
        sub_tag = long_tags[sub_start: sub_end]

        if len(sub_sen) != len(sub_tag):
            print(sub_sen)
            print(sub_tag)
            print('A 子句和其标签数量不对等')
            continue

        long_sub_sens.append(' '.join(list(sub_sen)))
        long_sub_tags.append(sub_tag)
        temp_end = sub_end

    # 截取天气预报格式文本最后的一部分文本
    if temp_end != last_postion:
        sub_sen = long_text[temp_end: last_postion]
        sub_tag = long_tags[temp_end: last_postion]

        if len(sub_sen) != len(sub_tag):
            print(sub_sen)
            print(sub_tag)
            print('B 子句和其标签数量不对等')
        long_sub_sens.append(' '.join(list(sub_sen)))
        long_sub_tags.append(sub_tag)


    return long_sub_sens, long_sub_tags


def text_to_dataset(data_type='train'):

    file_path = {'train': 'msra/train/', 'test': 'msra/test/', 'valid': 'msra/val/'}
    sentences = codecs.open(file_path[data_type] + 'sentences.txt')
    tags = codecs.open(file_path[data_type] + 'tags.txt')
    tag_to_index = {tag.strip(): index for index, tag in enumerate(codecs.open('msra/tags.txt'))}

    # 存储长度合适的样本
    norm_sens, norm_tags = [], []
    # 存储长度不合适的样本
    long_sens, long_tags = [], []
    for sen, tag in zip(sentences, tags):
        sen = sen.strip().split()
        # tag 转换为数字表示
        tag = tag.strip().split()
        tag = [tag_to_index[t] for t in tag]

        sen_len = len(sen)
        tag_len = len(tag)

        # 数据集不存在这种情况，这里做一个检查
        if sen_len != tag_len:
            continue

        if sen_len <= 500:
            norm_sens.append(' '.join(sen))
            norm_tags.append(tag)
            continue


        if data_type in ['train', 'valid']:
            long_sub_sens, long_sub_tags = split_train_test_long_text(''.join(sen), tag)
        else:
            # 将长度超过 500 的句子单独存储，后续单独处理
            long_sens.append(''.join(sen))
            long_tags.append(tag)


    # 数据存储
    Dataset.from_dict({'sentence': norm_sens, 'label': norm_tags}).save_to_disk('data/' + data_type + '.data')
    Dataset.from_dict({'sentence': long_sens, 'label': long_tags}).save_to_disk('data/' + data_type + '.long')


def data_process():
    text_to_dataset(data_type='train')
    text_to_dataset(data_type='test')
    text_to_dataset(data_type='valid')

    # 合并训练集和验证集
    train_data = load_from_disk('data/train.data')
    valid_data = load_from_disk('data/valid.data')

    train_valid_data = concatenate_datasets([train_data, valid_data])
    train_valid_data.save_to_disk('data/train_valid.data')


def show_length():
    for path in glob.glob('data/*.data'):
        all_data = load_from_disk(path)
        print(path, all_data)
        for data in all_data:
            if len(data['sentence'].split()) > 500:
                print('长度:', data['sentence'])
            if len(data['sentence'].split()) != len(data['label']):
                print('匹配:', data['sentence'])


# 测试集中的 4 条长文本没有进行切割，跳过
if __name__ == '__main__':
    data_process()
    show_length()