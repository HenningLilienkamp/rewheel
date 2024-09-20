from abc import ABC, abstractmethod
import idx2numpy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class AbstractDataReader(ABC):

    @abstractmethod
    def get_data(self):
        pass

class MNISTReader(AbstractDataReader):
    def __init__(self):
        self.__ident = 'MNISTReader'

    def get_data(self, image_file, label_file, norm_image=True):
        X = idx2numpy.convert_from_file(image_file)
        Xmin = np.min(X)
        Xmax = np.max(X)
        
        if norm_image:
            X = (X - Xmin) / (Xmax - Xmin)

        y = idx2numpy.convert_from_file(label_file).reshape(-1, 1)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(y)
        y_1hot = enc.transform(y).toarray()
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y_1hot,
                                                        test_size=0.33,
                                                        random_state=42)
        

        return X_train, X_test, y_train, y_test, Xmin, Xmax

    