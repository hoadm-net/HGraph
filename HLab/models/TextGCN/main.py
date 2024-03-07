from sklearn import datasets
from HLab.hmd.preprocessing import *
from HLab.hmd import Utilities as Util
from lib import NGDataset
from os import path


if __name__ == '__main__':
    data, labels = datasets.fetch_20newsgroups(
        data_home=Util.get_data_path('20newsgroups'),
        subset='all',
        return_X_y=True
    )

    print(data[0])
    
    preprocessor = StringPreprocessing(data[0])
    preprocessor.add_handler(ToLowerCase())
    
    preprocessor.add_handler(RemoveWhiteSpace())
    preprocessor.add_handler(RemovePunctuation())

    output = preprocessor.execute()

    print(output)

    # data = NGDataset()
    # data.process()
