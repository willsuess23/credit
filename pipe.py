# @Author: Will Suess
# @Date:   2018-06-07 20:26:35 PM
# @Email:  will.suess@cfraresearch.com
# @Last modified by:   Will Suess
# @Last modified time: 2018-06-08 05:33:41 AM

"""
Classes That Help Transition between Pandas and Sklearn
"""
from sklearn.base import TransformerMixin, BaseEstimator

class FactorExtractor(TransformerMixin,BaseEstimator):
    def __init__(self,factor):
        self.factor = factor

    def transform(self,data):
        return data.loc[:,self.factor]

    def fit(self,*_fit):
        return self
