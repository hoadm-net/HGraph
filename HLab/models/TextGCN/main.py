from sklearn import datasets
from HLab.hmd.preprocessing import *
from HLab.hmd import Utilities as Util
from classes import NGDataset


if __name__ == '__main__':
    inp = [
        'Chợ thịt chó nổi tiếng ở Sài Gòn bị truy quét',
        'Giá vàng miếng và nhẫn trơn đồng loạt phá kỷ lục'
    ]
    
    preprocessor = StringPreprocessing()
    preprocessor.add_handler(ToLowerCase())
    preprocessor.add_handler(RemoveWhiteSpace())
    preprocessor.add_handler(RemovePunctuation())
    preprocessor.add_handler(VietnameseTokenizer())

    for i in inp:
        o = preprocessor.execute(i)
        print(o)

    # data = NGDataset()
    # data.process()
