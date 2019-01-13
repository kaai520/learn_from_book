from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline #串行算法的管道

def polynomial_model(degree=1):
    # include_bias 为True也就是说有偏置项，interaction_only=False也就是有x1^2这种自乘项
    poly=PolynomialFeatures(degree=degree, include_bias=True,interaction_only=False)
    #正则化了
    linear_regession=LinearRegression(normalize=True)
    pipeline=Pipeline([('poly',poly),('lr',linear_regession)])
    return pipeline
