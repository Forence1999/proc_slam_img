import itertools
import os
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import *
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import _check_sample_weight
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from _ransac import RANSACRegressor
from sklearn.cluster import DBSCAN, KMeans
from scipy import stats
from copy import deepcopy
from scipy.spatial.distance import cdist

from PIL import Image


# class POINT():
#     def __init__(self, id, coordinates, neighbors=None, num_neighbor=None, is_noise=None, is_core=None):
#         super(POINT, self).__init__()
#         self.id = id
#         self.coord = coordinates
#         self.neighbors = neighbors
#         self.num_neighbor = num_neighbor
#         self.is_noise = is_noise
#         self.is_core = is_core
#
#     def get_id(self):
#         return self.id
#

class DBSCAN_4_line_detection(ClusterMixin, BaseEstimator):
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=5, p=None, n_jobs=None):
        '''
        :param eps:
        :param min_samples:
        :param metric:
        :param metric_params:
        :param algorithm:
        :param leaf_size:
        :param p: The power of the Minkowski metric
        :param n_jobs:
        '''
        super(DBSCAN_4_line_detection, self).__init__()
        self.wall_width = 4
        self.wall_width_eps = self.wall_width // 2 + 2
        self.walker_width = 13
        self.add_interval = self.wall_width * self.walker_width * 3
        self.wall_length_eps = self.walker_width
        self.float_eps = np.finfo(np.float32).eps
        self.dbscan_core_min_samples = min_samples
        self.dbscan = DBSCAN(eps=self.wall_length_eps, min_samples=self.dbscan_core_min_samples, metric='euclidean', )
        self.nearest_wide_distance2_per_line_pair = (self.wall_width_eps + 2) ** 2
        self.nearest_length_distance2_per_line_pair = (self.wall_length_eps + 5) ** 2
        self.least_degree_diff = 10
        
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs
        # self.unvisited = None
        # self.visited = None
        # self.point_ls = None
        self.neighborhoods = None
        self.num_neighbor = None
        # self.core_samples = None
        self.X = None
        self.labels_ = None  # -1: noise  |  0: unvisited core point  |  others: class
        # self.core_sample_indices_ = None
        self.__load_data__()
    
    def __load_data__(self):
        img_name = 'skeleton_0'
        img = cv2.imread('../image/skeleton/0.jpg', cv2.IMREAD_GRAYSCALE)
        self.figsize = np.array(img.shape)
        print('img_size: ', self.figsize)
        self.X = np.array(np.where(img == 255)).T
        self.npz_basename = '_'.join(('wall_w', str(round(self.wall_width_eps, 1)), 'wall_l',
                                      str(round(self.wall_length_eps, 1)),))
        self.npz_dir = os.path.join('../npz', img_name)
        os.makedirs(self.npz_dir, exist_ok=True)
    
    def preprocessing(self, X, y=None, sample_weight=None):
        X = self._validate_data(X, accept_sparse='csr')
        
        if not self.wall_length_eps > 0.0:
            raise ValueError("wall_length_eps must be positive.")
        
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
        
        # Calculate neighborhood for all samples. This leaves the original
        # point in, which needs to be considered later (i.e. point i is in the
        # neighborhood of point i. While True, its useless information)
        if self.metric == 'precomputed' and sparse.issparse(X):
            # set the diagonal to explicit values, as a point is its own
            # neighbor
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
                X.setdiag(X.diagonal())  # XXX: modifies X's internals in-place
        
        neighbors_model = NearestNeighbors(
            radius=self.wall_length_eps, algorithm=self.algorithm, leaf_size=self.leaf_size,
            metric=self.metric, metric_params=self.metric_params, p=self.p, n_jobs=self.n_jobs)
        neighbors_model.fit(X)
        # This has worst case O(n^2) memory complexity
        neighborhoods = neighbors_model.radius_neighbors(X, return_distance=False)
        
        if sample_weight is None:
            n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])
        else:
            n_neighbors = np.array([np.sum(sample_weight[neighbors]) for neighbors in neighborhoods])
        
        # Initially, all samples are noise.
        # labels = np.full(X.shape[0], -1, dtype=np.intp)
        
        # A list of all core samples found.
        core_samples = np.asarray(n_neighbors >= self.dbscan_core_min_samples, dtype=np.uint8)
        # point_ls = []
        # for i in range(len(X)):
        #     point_ls.append(
        #         POINT(id=i, coordinates=X[i], neighbors=neighborhoods[i], num_neighbor=n_neighbors[i], is_noise=True,
        #               is_core=core_samples[i]))
        self.neighborhoods = neighborhoods
        self.num_neighbor = n_neighbors
        self.labels_ = np.array(core_samples, dtype=np.int16) - 1
        # self.core_samples = core_samples
        # self.core_sample_indices_ = np.where(core_samples)[0]
        # self.point_ls = point_ls
        # self.unvisited = set(range(len(X)))
        # self.visited = set()
        self.X = X
    
    def _RANSACRegressor(self, x, y, thin_line=False):
        if thin_line:
            def RANSAC_loss(y, y_pred):
                ls = np.array(np.abs(y - y_pred) > self.wall_width_eps, dtype=np.float32)
                return ls.reshape(-1, )
        else:
            def RANSAC_loss(y, y_pred):
                ls = np.array(np.abs(y - y_pred) > self.wall_width_eps, dtype=np.float32)  # self.wall_width / 2
                return ls.reshape(-1, )
        
        # 拟合 is_data_valid=is_data_valid,
        ransac = RANSACRegressor(min_samples=2, residual_threshold=0.5, max_trials=100, loss=RANSAC_loss, )
        # residual_threshold 为根据loss值划分inlier和outlier的阈值，由于使用了0、1损失。故只需设置为（0，1）区间内，便有同样的效果，无需调整
        ransac.fit(x, y)
        # ransac.fit(points, np.random.randn(num_data))
        
        return {
            'inlier_mask'  : ransac.inlier_mask_,
            'estimator_'   : ransac.estimator_,
            'is_transposed': ransac.is_transposed,
        }
    
    def split_cluster(self, cluster, fail_return_None=False):
        # print('Length of cluster: ', len(cluster))
        if len(cluster) <= 2:
            return cluster, set()
        
        points = self.X[list(cluster)]
        x = points[:, 0].reshape(-1, 1)
        y = points[:, 1].reshape(-1, 1)
        
        try:
            RANSAC_res = self._RANSACRegressor(x, y, )
            inlier_mask = RANSAC_res['inlier_mask']
            prospective_cluster = points[inlier_mask]
            
            # 判断连通性,取点数最大的集合作为筛选之后的cluster
            y_pred = self.dbscan.fit_predict(prospective_cluster)
            most_cls = stats.mode(y_pred)[0][0]
            most_points = prospective_cluster[np.where(y_pred == most_cls)[0]]
            
            true_cluster = set()
            for i in most_points:
                true_cluster.add(np.where(np.all(self.X == i, axis=-1))[0][0])
            unavailable = set.difference(cluster, true_cluster)
            
            return true_cluster, unavailable
        
        except:
            print('RANSACRegressor failed to split the cluster!')
            if fail_return_None:
                return None
            else:
                unavailable = deepcopy(cluster)
                true_cluster = set()
                true_cluster.add(unavailable.pop())
                true_cluster.add(unavailable.pop())
                
                return true_cluster, unavailable
    
    def update_candidate_set(self, cluster, unavailable_point):
        candidate_set = set()
        cluster_ls = list(cluster)
        for i in cluster_ls:
            neighbors = self.neighborhoods[i]
            for j in neighbors:
                if (self.labels_[j] <= 0) and (j not in unavailable_point) and (j not in cluster):
                    candidate_set.add(j)
        return candidate_set
    
    def calculate_nearest_distance_between_2_clusters(self, cluster_1, cluster_2, ):
        # cluster_1, cluster_2 = np.array(cluster_1), np.array(cluster_2)
        # combine = [(i, j) for i in cluster_1 for j in cluster_2]
        # distance = [np.linalg.norm(i - j) for (i, j) in combine]
        # return min(distance)
        
        distance = cdist(cluster_1, cluster_2, metric='cityblock')
        
        return distance.min()
    
    def combine_cluster(self):
        ipath = os.path.join(self.npz_dir, 'dbscan_' + self.npz_basename + '.npz')
        data = np.load(ipath, )
        self.X = data['X']
        self.labels_ = data['labels']
        update_flag = True
        num_point_threshold = 512
        while update_flag:
            update_flag = False
            clses = np.array(list(set(self.labels_)))
            cls_lengths = np.array([len(np.where(self.labels_ == cls)[0]) for cls in clses])
            clses = clses[cls_lengths >= num_point_threshold]
            
            combine_cls = list(itertools.combinations(clses, r=2))
            distance = []
            for i in combine_cls:
                cls_1, cls_2 = i
                cluster_1 = self.X[self.labels_ == cls_1]
                cluster_2 = self.X[self.labels_ == cls_2]
                distance.append(self.calculate_nearest_distance_between_2_clusters(cluster_1, cluster_2))
            combinations = list(zip(combine_cls, distance))
            combinations = sorted(combinations, key=lambda i: i[-1])
            for i in combinations:
                (cls_1, cls_2), distance = i
                if distance > self.walker_width:
                    break
                # cls_1, cls_2 = i
                set_1 = set(np.where(self.labels_ == cls_1)[0])
                set_2 = set(np.where(self.labels_ == cls_2)[0])
                len_1, len_2 = len(set_1), len(set_2)
                set_12 = set.union(set_1, set_2, )
                split_res = self.split_cluster(set_12, fail_return_None=True)
                if split_res is None:
                    continue
                new_set_1, unavailable = split_res
                split_res = self.split_cluster(unavailable, fail_return_None=True)
                if split_res is None:
                    new_set_2 = set()
                    unavailable = set.difference(set_12, new_set_1)
                else:
                    new_set_2, unavailable = split_res
                new_len_1, new_len_2 = len(new_set_1), len(new_set_2)
                len_max, new_len_max = max([len_1, len_2]), max([new_len_1, new_len_2])
                if len_max < new_len_max:
                    if new_len_max - len_max >= 0:
                        update_flag = True
                    print('len_1 - len_2 - new_len_1 - new_len_2 - unavailable: ',
                          len_1, len_2, new_len_1, new_len_2, len(unavailable))
                    self.labels_[list(new_set_1)] = cls_1
                    self.labels_[list(new_set_2)] = cls_2
                    if len(unavailable) > 0:
                        un_labels = self.dbscan.fit_predict(self.X[list(unavailable)]) + np.max(self.labels_) + 2
                        self.labels_[list(unavailable)] = un_labels
                    break
            if (not update_flag) and num_point_threshold > 4:
                update_flag = True
                num_point_threshold //= 2
                print('num_point_threshold: ', num_point_threshold)
        opath = os.path.join(self.npz_dir, 'combined_dbscan_' + self.npz_basename + '.npz')
        np.savez(opath, X=self.X, labels=self.labels_)
    
    def project_point_2_line(self, point, line):
        # calculate the distance form p to ab
        #                 p
        #                 |
        #                 |
        # a---------b ----c
        
        p1, p2 = point
        a1, a2, b1, b2 = line
        ab = np.array([b1 - a1, b2 - a2])
        ap = np.array([p1 - a1, p2 - a2])
        
        ac = (ap @ ab) * ab / (ab @ ab + self.float_eps)
        cp = ap - ac
        
        return np.array((a1 + ac[0], a2 + ac[1])), cp
    
    def calculate_candidate_new_points(self, cluster_1, cluster_2):
        x1, y1 = cluster_1[:, 0], cluster_1[:, 1],
        x2, y2 = cluster_2[:, 0], cluster_2[:, 1],
        
        RANSAC_1 = self._RANSACRegressor(x1, y1)
        estimator_1 = RANSAC_1['estimator_']
        inlier_mask_1 = RANSAC_1['inlier_mask']
        # is_transposed_1 = RANSAC_1['is_transposed']
        k1, b1 = estimator_1.coef_, estimator_1.intercept_
        
        theta = np.arctan(k1)
        
        if k1 == np.inf:
            end_points = np.array([[-b1, np.min(y1)], [-b1, np.max(y1)]])
        else:
            
            (prj_min_x1, prj_min_y1), _ = self.project_point_2_line((np.min(x1), np.min(x1) * k1 + b1),
                                                                    line=(2, 2 * k1 + b1, 5, 5 * k1 + b1,))
            (prj_max_x1, prj_max_y1), _ = self.project_point_2_line((np.max(x1), np.max(x1) * k1 + b1),
                                                                    line=(2, 2 * k1 + b1, 5, 5 * k1 + b1,))
            
            end_points = np.array([[prj_min_x1, prj_min_y1], [prj_max_x1, prj_max_y1]])
        x_seg_len = self.wall_width // 2 * 2 + 1
        y_seg_len = self.walker_width
        new_points_offset = np.array([[i, j]
                                      for i in (np.arange(x_seg_len) - x_seg_len // 2)
                                      for j in np.arange(y_seg_len)])
        
        # x_seg_len = np.max(round(self.walker_width * np.cos(np.arctan(abs(k1)))),
        #                    y_seg_len=round(self.wall_width * np.cos(np.arctan(abs(k1)))) // 2 * 2 + 1
        # new_points_offset = np.array([[i, j]
        #                               for i in (np.arange(x_seg_len))
        #                               for j in np.arange(y_seg_len) - y_seg_len // 2])
        for i in range(2):
            new_points = np.around(end_points[i] + new_points_offset, dtype=np.int32)
        np.all()
        
        RANSAC_2 = self._RANSACRegressor(x2, y2)
        estimator_2 = RANSAC_2['estimator_']
        
        pass
    
    def extend_line_2_combine_cluster(self):
        
        data = np.load('./combined_dbscan_4.npz', )
        
        self.X = data['X']
        self.labels_ = data['labels']
        
        new_points = set()
        num_point_threshold = 16
        clses = np.array(list(set(self.labels_)))
        cls_lengths = np.array([len(np.where(self.labels_ == cls)[0]) for cls in clses])
        extend_clses = clses[cls_lengths >= num_point_threshold]
        cls_combinations = [[i, j] for i in extend_clses for j in clses]
        for (cls_1, cls_2) in cls_combinations:
            cluster_1, cluster_2 = self.X[self.labels_ == cls_1], self.X[self.labels_ == cls_2]
            distance = self.calculate_nearest_distance_between_2_clusters(cluster_1, cluster_2)
            if distance > self.walker_width * 2 ** 0.5:
                continue
            candidate_new_points = self.calculate_candidate_new_points(cluster_1, cluster_2)
            new_points.update(candidate_new_points)
        
        np.savez('./extended_combined_dbscan_4.npz', X=self.X, labels=self.labels_, new_points=new_points)
    
    def calculate_lengths(self, lines):
        try:
            lines = np.array(lines)
            point1, point2 = lines[:, :2], lines[:, 2:]
            lengths = np.linalg.norm(point1 - point2, ord=2, axis=-1)
        except:
            print('wait')
            return [0]
        return lengths
    
    def calculate_degrees(self, lines):
        lines = np.array(lines)
        x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3],
        radian = np.zeros(shape=(len(lines),))
        close_idx = np.isclose(x2, x1, )
        radian[close_idx] = np.pi / 2
        radian[~ close_idx] = np.arctan((y2[~ close_idx] - y1[~ close_idx]) / (x2[~ close_idx] - x1[~ close_idx]))
        degree = radian / np.pi * 180.
        return degree
    
    def calculate_max_group_degree_diff(self, degrees):
        # 斜率描述的直线角度（-pi/2, pi/2）, 同时认为 -pi/2与pi/2很近
        if len(degrees) == 1:
            return 0.
        degree_diff = [degrees[i] - degrees[j]
                       for i in range(len(degrees)) for j in range(i + 1, len(degrees))]
        degree_diff = 90. - np.abs(90. - np.abs(degree_diff))
        return degree_diff.max()
    
    def group_degrees(self, degrees, ):
        
        def map_degree_to_circle_coordinates(degrees):
            circle_coordinates = []
            for i in degrees:
                i = i * 2 / 180. * np.pi  # * 2 将(-pi/2,pi/2)映射为(-pi, pi)
                circle_coordinates.append([np.cos(i), np.sin(i)])
            return np.array(circle_coordinates)
        
        ### return idx
        degrees = np.array(degrees)
        ini_groups = [list(range(len(degrees)))]
        res_groups = []
        
        while len(ini_groups) > 0:
            group_idx = np.array(ini_groups.pop())
            group_degree = degrees[group_idx]
            if (0 < len(group_idx) <= 1) or self.calculate_max_group_degree_diff(group_degree) < self.least_degree_diff:
                res_groups.append(group_idx)
                continue
            
            coordinates = map_degree_to_circle_coordinates(group_degree)
            clf = KMeans(n_clusters=2)
            clf.fit(coordinates)
            # centers = clf.cluster_centers_  # 数据中心点
            clf_label = clf.labels_
            for i in range(2):
                idx = group_idx[clf_label == i]
                if len(idx) > 0:
                    ini_groups.append(idx)
        print('Number of groups: ', len(res_groups))
        
        for group in res_groups:
            degree = degrees[group]
            print('degrees: ', degree)
            print('max_degree_diff: ', self.calculate_max_group_degree_diff(degree))
        
        # for i in range(len(res_groups)):
        #     line, degree = list(zip(*(res_groups[i])))
        #     for j in degree:
        #         plt.scatter(j[0], j[0], c=['r', 'g', 'b', ][i % 3])
        #     plt.scatter(np.mean(degree), np.mean(degree), marker='*', s=100)
        # plt.show()
        return res_groups
    
    def combine_lines_per_group(self, group):
        lines, _ = list(zip(*group))
        lines = np.array(lines, dtype=np.float32)
        while True:
            merge_flag = 0
            lengths = self.calculate_lengths(lines)
            group = list(zip(lines, lengths))
            group = sorted(group, key=lambda record: record[-1], reverse=True, )
            lines, _ = list(zip(*group))
            lines = list(lines)
            for i in range(len(lines)):
                l_line = lines[i]
                for j in range(i + 1, len(lines)):
                    s_line = lines[j]
                    merge_line = self.merge_lines_per_pair(l_line, s_line)
                    if merge_line is not None:
                        merge_flag = 1
                        del lines[j], lines[i]
                        lines.append(merge_line)
                        break
                if merge_flag == 1:
                    break
            if merge_flag == 0:
                break
        
        return np.array(lines)
    
    def calculate_distance2_point_2_line(self, point, line, details=False, return_nearest_point=False):
        # calculate the distance form p to ab
        #                 p
        #                 |
        #                 |
        # a---------b ----c
        p1, p2 = point
        a1, a2, b1, b2 = line
        ab = np.array([b1 - a1, b2 - a2])
        ap = np.array([p1 - a1, p2 - a2])
        bp = np.array([p1 - b1, p2 - b2])
        
        gamma = (ap @ ab) / (ab @ ab + self.float_eps)
        ac = gamma * ab
        cp = ap - ac
        
        if gamma <= 0:
            d_shortest = np.square(ap).sum()
        elif gamma >= 1:
            d_shortest = np.square(bp).sum()
        else:
            d_shortest = np.square(cp).sum()
        
        if details:
            return np.square(ap).sum(), np.square(bp).sum(), np.square(cp).sum(), d_shortest
        elif return_nearest_point:
            if gamma <= 0:
                return d_shortest, (a1, a2)
            elif gamma >= 1:
                return d_shortest, (b1, b2)
            else:
                return d_shortest, (a1 + ac[0], a2 + ac[1])
        else:
            return d_shortest
    
    def calculate_general_line_form(self, line, ):
        x1, y1, x2, y2 = line
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        return a, b, c
    
    def calculate_lines_cross_point(self, line1, line2, ):
        '''

        :param line1:
        :param line2:
        :param exist_parallel: False: Force to return the intersection point at infinity
        :return:
        '''
        a1, b1, c1 = self.calculate_general_line_form(line1)
        a2, b2, c2 = self.calculate_general_line_form(line2)
        D = a1 * b2 - a2 * b1
        if np.allclose(D, 0.):
            return None
        x = (b1 * c2 - b2 * c1) / D
        y = (a2 * c1 - a1 * c2) / D
        return (x, y)
    
    def calculate_distance2_per_line_pair(self, line1, line2):
        cross_point = self.calculate_lines_cross_point(line1, line2)
        
        if (cross_point is not None) and (self.calculate_distance2_point_2_line(cross_point, line1) <= 2) and (
                self.calculate_distance2_point_2_line(cross_point, line2) <= 2):  # line segments intersect
            return 0
        else:  # 两条线段平行和不相交的计算方式相同
            distances = []
            distances.append(self.calculate_distance2_point_2_line(line1[:2], line2))
            distances.append(self.calculate_distance2_point_2_line(line1[2:], line2))
            distances.append(self.calculate_distance2_point_2_line(line2[:2], line1))
            distances.append(self.calculate_distance2_point_2_line(line2[2:], line1))
            
            return min(distances)
    
    def calculate_lines_radian(self, line1, line2):
        ab1 = np.array(line1[:2]) - np.array(line1[2:])
        ab2 = np.array(line2[:2]) - np.array(line2[2:])
        cos_alpha = (ab1 @ ab2) / (np.linalg.norm(ab1, ord=2) * np.linalg.norm(ab2, ord=2))
        alpha = np.arccos(cos_alpha)
        # alpha = np.arccos(np.sign(cos_alpha) * (abs(cos_alpha) - self.eps))
        
        return alpha
    
    def rotate_point(self, center, point, radian):  ### 顺时针旋转
        cx, cy, px, py = *center, *point
        x = (px - cx) * np.cos(radian) - (py - cy) * np.sin(radian) + cx
        y = (py - cy) * np.cos(radian) + (px - cx) * np.sin(radian) + cy
        return (x, y)
    
    def rotate_line(self, center, line, radian):
        a = self.rotate_point(center, line[:2], radian)
        b = self.rotate_point(center, line[2:], radian)
        return np.concatenate((a, b))
    
    def rotate_lines(self, center, lines, radian):
        rotated_lines = []
        for line in lines:
            rotated_lines.append(self.rotate_line(center, line, radian))
        return rotated_lines
    
    def merge_lines_per_pair(self, line1, line2):
        distance2 = self.calculate_distance2_per_line_pair(line1, line2)
        if distance2 <= self.nearest_wide_distance2_per_line_pair:  #
            ### combine
            # calculate Projection line
            # line1: a1 ------------- b1
            # line2:          a2 --------- b2    d_as: shortest distance from line2_a to line1
            # line : a ------------------- b
            
            length1, length2 = self.calculate_lengths([line1])[0], self.calculate_lengths([line2])[0]
            if length1 < length2:  # 长线在前
                line1, line2 = line2, line1
                length1, length2 = length2, length1
            # 若line1覆盖line2，则直接返回line1
            prj_point21, _ = self.project_point_2_line(point=line2[:2], line=line1)
            prj_point22, _ = self.project_point_2_line(point=line2[2:], line=line1)
            if (self.calculate_distance2_point_2_line(prj_point21, line1) <= 2) and (
                    self.calculate_distance2_point_2_line(prj_point22, line1) <= 2):
                return line1
            else:
                del prj_point21, prj_point22
            # line1与line2有投影也不重合的部分，即两线结合能够变长
            ratio = length2 / (length1 + length2)
            cross_point = self.calculate_lines_cross_point(line1, line2, )
            radian = self.calculate_lines_radian(line1, line2)
            alpha = np.pi / 2 - np.abs(np.pi / 2 - radian)
            if (cross_point is None):  # 线段所在直线平行
                _, p_vector1 = self.project_point_2_line(point=line2[:2], line=line1)
                _, p_vector2 = self.project_point_2_line(point=line2[2:], line=line1)
                p_vector = np.mean([p_vector1, p_vector2], axis=0)
                p_vector_1 = p_vector * ratio
                p_vector_2 = - p_vector * (1 - ratio)
                prj_points = [line1[:2] + p_vector_1, line1[2:] + p_vector_1, line2[:2] + p_vector_2,
                              line2[2:] + p_vector_2]
            else:  # 线段所在直线相交
                # print('cross_point: {:.2f} ---- {:.2f}'.format(*cross_point))
                angel = alpha * ratio
                line1_1 = self.rotate_line(cross_point, line1, angel)
                line1_2 = self.rotate_line(cross_point, line1, -angel)
                radian_1 = self.calculate_lines_radian(line1_1, line2)
                alpha_1 = np.pi / 2 - np.abs(np.pi / 2 - radian_1)
                radian_2 = self.calculate_lines_radian(line1_2, line2)
                alpha_2 = np.pi / 2 - np.abs(np.pi / 2 - radian_2)
                line = line1_1 if (alpha_1 <= alpha_2) else line1_2
                prj_point11, _ = self.project_point_2_line(point=line1[:2], line=line)
                prj_point12, _ = self.project_point_2_line(point=line1[2:], line=line)
                prj_point21, _ = self.project_point_2_line(point=line2[:2], line=line)
                prj_point22, _ = self.project_point_2_line(point=line2[2:], line=line)
                prj_points = [prj_point11, prj_point12, prj_point21, prj_point22]
                assert min(alpha_1, alpha_2) < alpha
            return self.find_longest_line_from_points(prj_points)
            
            # alpha = self.calculate_lines_radian(line1, line2)
            # alpha = np.pi / 2 - np.abs(np.pi / 2 - alpha)
            # angel = alpha * length2 / (length1 + length2)
            # print('Degree contract:', alpha * 180 / np.pi, angel * 180 / np.pi)
            # cross_point = self.calculate_lines_cross_point(line1, line2, )
            #
            # line = self.merge_point_line(line2[:2], line1)
            # line = self.merge_point_line(line2[2:], line)
            #
            # # rotate the line
            # line = self.calibrate_line(line1, line2, line)
            # return line
            
            # line1: a1 ------------- b1
            # line2:          a2 --------- b2    d_as: shortest distance from line2_a to line1
            # d_aa, d_ab, d_av, d_as = self.calculate_distance2_point_2_line(a2, line1, details=True)
            # d_ba, d_bb, d_bv, d_bs = self.calculate_distance2_point_2_line(b2, line1, details=True)
            # if np.isclose(d_as, d_av) and np.isclose(d_bs, d_bv):  # 两线重叠 ---- 最短距离==垂直距离，点在线旁，返回长线
            #     return line1
            # elif np.isclose(d_as, d_av) and (not np.isclose(d_bs, d_bv)):  # a在线旁，b在线外，b做投影，并将长线延长至投影点
            #     #TODO
            #
            # elif (not np.isclose(d_as, d_av)) and np.isclose(d_bs, d_bv):  # a在线外，b在线旁
            # # TODO
            # else:  # a、b均在线外，两线相距
            # # TODO
        else:
            return None
    
    def find_longest_line_from_points(self, points):
        line_ls = list(itertools.combinations(points, r=2))
        line_ls = [np.concatenate(i) for i in line_ls]
        idx = np.argmax(self.calculate_lengths(line_ls))
        
        return line_ls[idx]
    
    def merge_group_lines(self, group_lines, ):
        merge_groups = []
        for g_i, g_lines in enumerate(group_lines):
            g_lengths = self.calculate_lengths(g_lines)
            combinations = list(zip(g_lines, g_lengths))
            
            merge_flag = True
            while merge_flag:
                merge_flag = False
                combinations = sorted(combinations, key=lambda i: i[-1], reverse=True)
                for i, j in itertools.combinations(range(len(combinations)), r=2):
                    line_1, length_1 = combinations[i]
                    line_2, length_2 = combinations[j]
                    merge_res = self.merge_lines_per_pair(line_1, line_2)
                    if merge_res is None:
                        continue
                    else:
                        del combinations[j], combinations[i]
                        combinations.append([merge_res, self.calculate_lengths([merge_res])[0]])
                        merge_flag = True
                        break
            merge_groups.append(list(zip(*combinations))[0])
        return merge_groups
    
    def connect_lines(self, lines):
        lines = list(lines)
        update_flag = True
        new_lines = []
        while update_flag:
            update_flag = False
            combinations = list(itertools.combinations(range(len(lines)), r=2))
            for (i, j) in combinations:
                line1, line2 = lines[i], lines[j]
                distance2 = self.calculate_distance2_per_line_pair(line1, line2)
                if distance2 > self.nearest_length_distance2_per_line_pair or distance2 < 2:
                    continue
                radian = self.calculate_lines_radian(line1, line2)
                degree = radian / np.pi * 180.
                v_degree = np.abs(degree - 90)  # 距离垂直的偏差角
                if v_degree < self.least_degree_diff:  # 线段所在直线近乎垂直
                    cross_point = self.calculate_lines_cross_point(line1, line2, )
                    points1 = [line1[:2], line1[2:], cross_point]
                    update_line1 = self.find_longest_line_from_points(points1)
                    points2 = [line2[:2], line2[2:], cross_point]
                    update_line2 = self.find_longest_line_from_points(points2)
                    
                    update_flag = True
                    del lines[j], lines[i]
                    lines.append(update_line1)
                    lines.append(update_line2)
                    break
                
                else:  # 线段所在直线不垂直
                    length1, length2 = self.calculate_lengths([line1, line2])
                    if length1 < length2:  # 长线在前
                        length1, length2 = length2, length1
                        line1, line2 = line2, line1
                    # 线段所在直线近乎平行
                    if (90. - v_degree) < self.least_degree_diff:
                        prj_point21, _ = self.project_point_2_line(point=line2[:2], line=line1)
                        prj_point22, _ = self.project_point_2_line(point=line2[2:], line=line1)
                        if (self.calculate_distance2_point_2_line(prj_point21, line1) <= 2) or (
                                self.calculate_distance2_point_2_line(prj_point22, line1) <= 2):  # 有重合，和不平行不垂直的处理方式相同
                            pass
                        else:  # 无重合
                            dis_1 = self.calculate_distance2_point_2_line(line2[:2], line1)
                            dis_2 = self.calculate_distance2_point_2_line(line2[2:], line1)
                            point = line2[:2] if dis_1 < dis_2 else line2[2:]
                            _, prj_dis = self.project_point_2_line(point, line1)
                            prj_dis = np.square(prj_dis).sum()
                            if prj_dis <= self.nearest_wide_distance2_per_line_pair:
                                points = [line1[:2], line1[2:], line2[:2], line2[2:], ]
                                update_line = self.find_longest_line_from_points(points)
                                
                                update_flag = True
                                del lines[j], lines[i]
                                lines.append(update_line)
                                break
                    
                    # 线段所在直线不垂直不平行，有较大夹角
                    def calculate_connect_line(point, line):
                        distance2, nearest_point = self.calculate_distance2_point_2_line(point, line,
                                                                                         return_nearest_point=True)
                        if distance2 <= self.nearest_length_distance2_per_line_pair:
                            return (*point, *nearest_point)
                        else:
                            return None
                    
                    new_line = calculate_connect_line(line1[:2], line2)
                    if new_line is not None:
                        new_lines.append(new_line)
                    new_line = calculate_connect_line(line1[2:], line2)
                    if new_line is not None:
                        new_lines.append(new_line)
                    new_line = calculate_connect_line(line2[:2], line1)
                    if new_line is not None:
                        new_lines.append(new_line)
                    new_line = calculate_connect_line(line2[2:], line1)
                    if new_line is not None:
                        new_lines.append(new_line)
        lines.extend(new_lines)
        return np.array(lines, )
    
    def merge_lines(self):
        ipath = os.path.join(self.npz_dir, 'combined_dbscan_' + self.npz_basename + '.npz')
        data = np.load(ipath)
        self.X = data['X']
        self.labels_ = data['labels']
        clses = np.array(list(set(self.labels_)))
        cls_lengths = np.array([len(np.where(self.labels_ == cls)[0]) for cls in clses])
        clusters = [self.X[self.labels_ == cls] for cls in clses]
        lines = self.calculate_line_4_clusters(clusters)
        line_lengths = self.calculate_lengths(lines)
        line_lengths = np.array(line_lengths)
        ### 依据角度对线进行分组
        lines = np.array(lines)[line_lengths > 0]
        degrees = self.calculate_degrees(lines)
        groups_idx = self.group_degrees(degrees)
        group_lines = [lines[i] for i in groups_idx]
        ### 对角度相近的线进行合并
        group_lines = self.merge_group_lines(group_lines)
        lines = np.concatenate(group_lines, axis=0, )
        ### 对距离相近的线进行连接
        lines = self.connect_lines(lines)
        ### 旋转线段
        rotate_radian = self.calculate_rotate_radian(lines, )  # 逆时针旋转
        lines = self.rotate_lines(np.array(self.figsize) // 2, lines, radian=-rotate_radian)
        lines = np.array(lines, dtype=int)
        ### 绘出图像
        img = np.zeros((self.figsize[1], self.figsize[0]))
        for line in lines:
            if self.calculate_lengths([line])[0] > 0:
                cv2.line(img, line[:2], line[2:], 255, )
        ### 旋转图像
        
        # img = self.rotate_img(img, rotate_radian * 180. / np.pi)
        ### 保存并显示图片
        opath = os.path.join(self.npz_dir, self.npz_basename + 'rad' + str(rotate_radian) + '_line.jpg')
        cv2.imwrite(opath, np.flip(img, axis=0))
        cv2.imshow(self.npz_basename, np.flip(img, axis=0))
        cv2.waitKey(0)
    
    def rotate_img(self, image, degree, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, degree, scale)
        rotated_img = cv2.warpAffine(image, M, (w, h))
        return rotated_img
    
    def calibrate_img(self):
        ipath = os.path.join(self.npz_dir, 'combined_dbscan_' + self.npz_basename + '.npz')
        data = np.load(ipath)
        self.X = data['X']
        self.labels_ = data['labels']
        clses = np.array(list(set(self.labels_)))
        cls_lengths = np.array([len(np.where(self.labels_ == cls)[0]) for cls in clses])
        clusters = [self.X[self.labels_ == cls] for cls in clses]
        combination = list(zip(clusters, clses, cls_lengths))
        combination = sorted(combination, key=lambda i: i[-1], reverse=True)
        
        lines = self.calculate_line_4_clusters([*list(zip(*(combination[:5])))][0])
        rotate_radian = self.rotate_img(lines, )
        clusters = self.rotate_clusters(clusters, rotate_radian)
        
        combination = list(zip(clusters, clses, cls_lengths))
        combination = [i for i in combination if i[-1] > 0]
        lines = self.calculate_line_4_clusters([*list(zip(*(combination)))][0])
        # rotate_radian = -self.rotate_img(lines, )
        
        img = np.zeros((self.figsize[1], self.figsize[0]))
        for line in lines:
            cv2.line(img, line[:2], line[2:], 255, )
        
        opath = os.path.join(self.npz_dir, self.npz_basename + '_line.jpg')
        cv2.imwrite(opath, np.flip(img, axis=0))
        cv2.imshow(self.npz_basename, np.flip(img, axis=0))
        cv2.waitKey(0)
    
    def rotate_clusters(self, clusters, radian):
        for i, cluster in enumerate(clusters):
            img = np.zeros(self.figsize)
            for j in cluster:
                img[j[0]][j[1]] = 255
            rotate_img = Image.fromarray(np.array(img, dtype=np.uint8))
            img = rotate_img.rotate(radian * 180. / np.pi)
            img = np.array(img)
            clusters[i] = np.array(np.where(img > 0)).T
        return clusters
    
    def calculate_line_4_clusters(self, clusters, ):
        # num_point_threshold = 16
        lines = []
        # plot_clses = clses[cls_lengths >= num_point_threshold]
        # line_img = np.zeros((self.figsize[1], self.figsize[0]))
        for i, cluster in enumerate(clusters):
            x = cluster[:, 0].reshape(-1, 1)
            y = cluster[:, 1].reshape(-1, 1)
            # cv_res = cv2.fitLine(np.array([[1, 0, ], [1, 1]]), distType=cv2.DIST_L1, param=0, reps=1e-2, aeps=1e-2)
            # print('cv_res: ', np.round(cv_res.reshape(-1, ), 3))
            # print('k: ', cv_res[1] / cv_res[0])
            
            is_transposed = False
            if (x.max() - x.min()) < (y.max() - y.min()):
                x, y = y, x
                is_transposed = True
            
            linreg = LinearRegression()
            linreg.fit(x, y)
            k, b = linreg.coef_[0][0], linreg.intercept_[0]
            # print('k - b: ', k, b)
            if is_transposed:
                line = (x.min() * k + b, x.min(), x.max() * k + b, x.max())
                # line = list(map(int, line))
            else:
                line = (x.min(), x.min() * k + b, x.max(), x.max() * k + b)
                # line = list(map(int, line))
            # print('line: ', line)
            lines.append(line)
            # cv2.line(line_img, line[:2], line[2:], 255, )
        # cv2.imshow('test', np.flip(line_img, axis=0))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return lines
    
    def calculate_rotate_radian(self, lines):
        ### 逆时针为正
        lengths = self.calculate_lengths(lines)
        combination = list(zip(lines, lengths))
        combination = sorted(combination, key=lambda i: i[-1], reverse=True)[:5]
        degrees = []
        # img = np.zeros((self.figsize[1], self.figsize[0]))
        for i, (line, length) in enumerate(combination):
            line = np.array(line)
            vector = line[2:] - line[:2]
            vector *= np.sign(vector[1])
            theta = np.arccos(vector @ np.array([1, 0]) / np.linalg.norm(vector))
            theta = theta % (np.pi / 2)
            rotate_theta = np.sign(theta - np.pi / 4) * (np.pi / 4 - np.abs(theta - np.pi / 4))
            degrees.append(-rotate_theta)
            
            # print('Vector: ', vector)
            # print('Theta: ', theta * 180. / np.pi)
            # print('Rotate_theta: ', np.around(rotate_theta, 3))
            #
            # cv2.line(img, line[:2], line[2:], 255, )
            # opath = os.path.join(self.npz_dir, self.npz_basename + '_line.jpg')
            # cv2.imwrite(opath, np.flip(img, axis=0))
            # cv2.imshow(self.npz_basename, np.flip(img, axis=0))
            # cv2.waitKey(0)
        
        print('Rotate_degrees: ', np.around(degrees, 3))
        print('Median degree: ', np.median(degrees))
        return np.mean(degrees)
    
    def plot_res(self, data, labels, figsize):
        color = ['r', 'lime', 'g', 'b', 'cyan', 'orange', 'magenta', 'blueviolet', 'navy']
        marker = ['o', 'v', '*', 'X', '^', '<', '>', 'p', 'P', 'd']
        cls = list(set(labels))
        plt.figure(figsize=figsize // 20, dpi=200)
        for i in range(len(cls)):
            idx = np.where(labels == cls[i])[0]
            points = data[idx]
            X = points[:, 0]
            Y = points[:, 1]
            plt.scatter(X, Y, marker=marker[i % len(marker)], color=color[i % len(color)], linewidths=0.01)
            # plt.text(X, Y, [str(i)] * len(Y), fontsize=0.1, verticalalignment="top", horizontalalignment="right")
        plt.legend(loc='upper right')
        
        opath = os.path.join(self.npz_dir, self.npz_basename + '.jpg')
        plt.savefig(opath)
        plt.show()
    
    def plot_cluster(self, cluster, X):
        data = X[list(cluster)]
        labels = [1] * len(data)
        self.plot_res(data, labels, self.figsize / 10)
    
    def fit(self, X, y=None, sample_weight=None):
        self.preprocessing(X=X, y=y, sample_weight=sample_weight)
        # 全局DBSCAN
        # y_pred = self.dbscan.fit_predict(self.X)
        # print('Global total number of clusters: ', len(set(y_pred.reshape(-1, ))))
        # plot_res(data=self.X, labels=y_pred.reshape(-1, ), figsize=self.figsize)
        # 分段DBSCAN
        crt_cls = 1
        num_core = len(np.where(self.labels_ == 0)[0])
        while num_core > 0:
            i = np.random.choice(np.where(self.labels_ == 0)[0])
            cluster = set()
            candidate_set = set(self.neighborhoods[i])
            unavailable_point = set()
            num_add = 0
            while len(candidate_set) > 0:
                j = candidate_set.pop()
                if (self.labels_[j] <= 0) and (j not in unavailable_point) and (j not in cluster):
                    num_add += 1
                    cluster.add(j)
                    for k in self.neighborhoods[j]:
                        if (self.labels_[k] <= 0) and (k not in unavailable_point) and (k not in cluster):
                            candidate_set.add(k)
                    if num_add > self.add_interval:
                        # plot_cluster(cluster, self.X)
                        cluster, unavailable = self.split_cluster(cluster)
                        unavailable_point.update(unavailable)
                        candidate_set = self.update_candidate_set(cluster, unavailable_point)
                        # candidate_set_update = self.update_candidate_set(cluster, unavailable_point)
                        # candidate_set.update(candidate_set_update)
                        num_add = 0
            cluster, unavailable = self.split_cluster(cluster)
            
            # print('-' * 50)
            # print('Length of cluster: ', len(cluster))
            # print('Number of core points: ', num_core)
            # print('Current class: ', crt_cls)
            self.labels_[list(cluster)] = crt_cls
            num_core = len(np.where(self.labels_ == 0)[0])
            # print('Number of core points: ', num_core)
            crt_cls += 1
        print('Total number of clusters: ', len(set(self.labels_)))
        
        path = os.path.join(self.npz_dir, 'dbscan_' + self.npz_basename + '.npz')
        np.savez(path, X=self.X, labels=self.labels_)
        return self
    
    def fit_predict(self, X=None, y=None, sample_weight=None):
        if X is None:
            X = self.X
        # self.fit(X, sample_weight=sample_weight)
        # self.combine_cluster()
        self.merge_lines()
        # self.calibrate_img()
        # self.plot_res(self.X, self.labels_, self.figsize)
        return self.labels_


def test_DBSCAN_4_line_detection():
    dbscan = DBSCAN_4_line_detection()
    dbscan.fit_predict()


if __name__ == '__main__':
    test_DBSCAN_4_line_detection()
