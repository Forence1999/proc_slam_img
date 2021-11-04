import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from sklearn.decomposition import PCA
from sklearn.linear_model import Lars, LinearRegression, RANSACRegressor, TheilSenRegressor
from copy import deepcopy
from sklearn.cluster import DBSCAN
from scipy import stats


def pca_fit_line(data):
    pca = PCA()
    pca.fit(data)
    print(pca.components_)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)


def LR_fit_line(data):
    linreg = LinearRegression()
    linreg.fit(data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1))
    print(linreg.coef_)
    print(linreg.intercept_)


# 直线方程函数
def scipy_fit_line(data):
    def f_1(x, A, B):
        return A * x + B
    
    x, y = data[:, 0], data[:, 1],
    # 拟合点
    A, B = optimize.curve_fit(f_1, x, y)[0]
    y_pred = A * x + B
    
    plot_res(x, y, y_pred, name='scipy')


def cv_fit_line(data):
    x, y = data[:, 0], data[:, 1],
    
    output = cv2.fitLine(data, distType=cv2.DIST_L1, param=0, reps=1e-2, aeps=1e-2)
    k = output[1] / output[0]
    ref_point = output[2:]
    y_pred = k * (x - ref_point[0]) + ref_point[1]
    
    plot_res(x, y, y_pred, name='cv2')


def sklearn_fit_line(data):
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    x_bkp = deepcopy(x)
    y_bkp = deepcopy(y)
    
    def loss(y, y_pred):
        ls = np.array(np.abs(y - y_pred) > 0.003, dtype=np.float32)
        return ls.reshape(-1, )
    
    # 拟合
    s0 = time.time()
    reg = RANSACRegressor(min_samples=3, residual_threshold=0.5, max_trials=100, max_skips=100, loss=loss,
                          stop_n_inliers=int(len(y) * 0.8), stop_probability=1., stop_score=np.inf)
    reg.fit(x, y)
    y_pred = reg.predict(x)
    s1 = time.time()
    plot_res(x, y, y_pred, name='RANSACRegressor')
    
    x, y = x_bkp, y_bkp
    y = y.reshape(-1, )
    s2 = time.time()
    reg = TheilSenRegressor()
    reg.fit(x, y)
    y_pred = reg.predict(x)
    s3 = time.time()
    # plot_res(x, y, y_pred, name='TheilSenRegressor')
    
    print('Time: ', s1 - s0, s3 - s2)


def plot_res(x, y, y_pred, name='test', x_pred=None):
    if x_pred is None:
        x_pred = x
    # 绘制散点
    plt.figure()
    plt.scatter(x, y, 1, "red")
    plt.plot(x_pred, y_pred, "blue")
    plt.title(name)
    plt.xlim(x.min() - 10, x.max() + 10)
    plt.ylim(y.min() - 10, y.max() + 10)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def split_cluster(data):
    def RANSAC_loss(y, y_pred):
        ls = np.array(np.abs(y - y_pred) > 4 / 2 + 1, dtype=np.float32)
        return ls.reshape(-1, )
    
    points = data
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1].reshape(-1, 1)
    
    # 拟合
    ransac = RANSACRegressor(min_samples=3, residual_threshold=0.5, max_trials=100, max_skips=100, loss=RANSAC_loss,
                             stop_n_inliers=int(len(y) * 0.8), stop_probability=1., stop_score=np.inf)
    # residual_threshold 为根据loss值划分inlier和outlier的阈值，由于使用了0、1损失。故只需设置为（0，1）区间内，便有同样的效果，无需调整
    ransac.fit(x, y)
    
    inlier_mask = ransac.inlier_mask_
    prospective_cluster = points[inlier_mask]
    dbscan = DBSCAN(eps=2 ** 0.5 + 0.0001, min_samples=1, metric='euclidean', )
    y_pred = dbscan.fit_predict(prospective_cluster)
    idx = stats.mode(y_pred)[0][0]
    cluster = prospective_cluster[y_pred == idx]
    plot_res(x, y, y_pred=cluster[:, 1], name='RANSACRegressor', x_pred=cluster[:, 0])


# 拟合直线
# if __name__ == '__main__':
#     data = np.array([[-1, 1], [0, 3], [1, 5], [2, 7], [3, 9], [-1, 2], ])
#     test_data = np.load('./test_data.npz')
#     # data = test_data['points']
#     img = test_data['img']
#     img[:, :180] = 0
#     img[:, 250:] = 0
#     # img[380:, :] = 0
#     data = np.array(np.where(img == 255)).T
#     # scipy_fit_line(data=data)
#     # cv_fit_line(data)
#     # sklearn_fit_line(data)
#     split_cluster(data)

# 测试PIL
if __name__ == '__main__':
    from PIL import Image
    
    img = Image.open("test.jpg")
    
    # 旋转方式一
    img1 = img.transpose(Image.ROTATE_180)  # 引用固定的常量值
    img1.save("r1.jpg")
    
    # 旋转方式二
    img2 = img.rotate(90)  # 自定义旋转度数
    img2 = img2.resize((400, 400))  # 改变图片尺寸
    img2.save("r2.jpg")
