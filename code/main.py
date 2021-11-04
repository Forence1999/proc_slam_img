import itertools
import os
from copy import deepcopy

import cv2
import numpy as np
from skimage import morphology
from sklearn.cluster import KMeans, DBSCAN


class MAP():
    def __init__(self, ini_img=None, ini_img_path=None, odir=None):
        '''
        The img_path has higher priority than the img.
        :param ini_img: initial image
        :param img_path: path of the initial image
        :param odir: dir of output
        '''
        super(MAP, self).__init__()
        if ini_img_path is not None:
            self.ini_img_path = ini_img_path
            ini_img = cv2.imread(self.ini_img_path, cv2.IMREAD_GRAYSCALE)
        if odir is None:
            odir = './image'
        self.count = 0
        
        self.odir = odir
        self.ini_img = ini_img  # TODO convert into Gray Scale Image
        self.img_name = os.path.basename(ini_img_path)
        self.bi_img = None
        self.binary_dir = os.path.join(self.odir, 'binary', )
        self.binary_path = os.path.join(self.binary_dir, self.img_name)
        self.de_img = None
        self.denoise_dir = os.path.join(self.odir, 'denoise', )
        self.denoise_path = os.path.join(self.denoise_dir, self.img_name)
        self.fill_img = None
        self.fill_gap_dir = os.path.join(self.odir, 'fill_gap', )
        self.fill_gap_path = os.path.join(self.fill_gap_dir, self.img_name)
        self.sk_img = None
        self.skeleton_dir = os.path.join(self.odir, 'skeleton', )
        self.skeleton_path = os.path.join(self.skeleton_dir, self.img_name)
        self.f_space_img = None
        self.free_space_dir = os.path.join(self.odir, 'free_space', )
        self.free_space_path = os.path.join(self.free_space_dir, self.img_name)
        self.line_img = None
        self.line_dir = os.path.join(self.odir, 'line', )
        self.line_path = os.path.join(self.line_dir, self.img_name)
        self.calibrate_img = None
        self.calibrate_dir = os.path.join(self.odir, 'calibrate', )
        self.calibrate_path = os.path.join(self.calibrate_dir, self.img_name)
        self.model_img = None
        self.model_map_dir = os.path.join(self.odir, 'model_map', )
        self.model_map_path = os.path.join(self.model_map_dir, self.img_name)
        self.seg_img = None
        self.segment_dir = os.path.join(self.odir, 'segment', )
        self.segment_path = os.path.join(self.segment_dir, self.img_name)
        self.graph_img = None
        self.graph_dir = os.path.join(self.odir, 'graph', )
        self.graph_path = os.path.join(self.graph_dir, self.img_name)
        self.test_img = None
        self.test_dir = os.path.join(self.odir, 'test', )
        self.test_path = os.path.join(self.test_dir, self.img_name)
        
        self.walker_width = 13
        self.eps = np.finfo(np.float32).eps
        self.nearest_distance2_per_line_pair = 4 ** 2
    
    def show_img(self, name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def show_all(self):
        # img_ls = []
        # img_name_ls = []
        #
        # if self.ini_img is not None:
        #     img_name_ls.append('ini')
        #     img_ls.append(self.ini_img)
        # if self.bi_img is not None:
        #     img_name_ls.append('binary')
        #     img_ls.append(self.bi_img)
        #
        # if self.de_img is not None:
        #     img_name_ls.append('denoise')
        #     img_ls.append(self.de_img)
        #
        # if self.fill_img is not None:
        #     img_name_ls.append('fill_gap')
        #     img_ls.append(self.fill_img)
        #
        # if self.sk_img is not None:
        #     img_name_ls.append('skeleton')
        #     img_ls.append(self.sk_img)
        #
        # if self.f_space_img is not None:
        #     img_name_ls.append('free_space')
        #     img_ls.append(self.f_space_img)
        #
        # if self.line_img is not None:
        #     img_name_ls.append('extract_lines')
        #     img_ls.append(self.line_img)
        #
        # if self.calibrate_img is not None:
        #     img_name_ls.append('calibrate_coordinate')
        #     img_ls.append(self.calibrate_img)
        #
        # if self.model_img is not None:
        #     img_name_ls.append('model_map')
        #     img_ls.append(self.model_img)
        #
        # if self.seg_img is not None:
        #     img_name_ls.append('segment')
        #     img_ls.append(self.seg_img)
        #
        # if self.graph_img is not None:
        #     img_name_ls.append('graph')
        #     img_ls.append(self.graph_img)
        #
        # num_img = len(img_ls)
        #
        # plt.figure(figsize=(3 * 50, int(np.ceil(num_img / 3)) * 50), dpi=80)
        # for i in range(num_img):
        #     plt.subplot(3, int(np.ceil(num_img / 3)), i + 1, )
        #
        #     plt.imshow(img_ls[i], 'gray')  #
        #     plt.title(img_name_ls[i])
        # plt.show()
        #
        # if self.ini_img is not None:
        #     cv2.imshow('ini', self.ini_img)
        # if self.bi_img is not None:
        #     cv2.imshow('binary', self.bi_img)
        if self.de_img is not None:
            cv2.imshow('denoise', self.de_img)
        if self.fill_img is not None:
            cv2.imshow('fill_gap', self.fill_img)
        if self.sk_img is not None:
            cv2.imshow('skeleton', self.sk_img)
        if self.f_space_img is not None:
            cv2.imshow('free_space', self.f_space_img)
        if self.line_img is not None:
            cv2.imshow('extract_lines', self.line_img)
        if self.calibrate_img is not None:
            cv2.imshow('calibrate_coordinate', self.calibrate_img)
        if self.model_img is not None:
            cv2.imshow('model_map', self.model_img)
        if self.seg_img is not None:
            cv2.imshow('segment', self.seg_img)
        if self.graph_img is not None:
            cv2.imshow('graph', self.graph_img)
        if self.test_img is not None:
            cv2.imshow('test', self.test_img)
        
        cv2.waitKey(300000)
        cv2.destroyAllWindows()
    
    def canny_img(self, img=None, opath=None):
        gauss_img = cv2.GaussianBlur(img, (5, 5), 0)
        cn_img = cv2.Canny(gauss_img, threshold1=150, threshold2=100)  # [, edges[, apertureSize[, L2gradient]]])
        self.test_img = cn_img
        self.sk_img = gauss_img
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, self.test_img)
        return self.test_img
    
    def binary_img(self, img=None, opath=None):
        threshold, bi_img, = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
        self.bi_img = bi_img
        # ['ini', 'THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV']
        
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, self.bi_img)
        return self.bi_img
    
    def denoise_img(self, img=None, opath=None):
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0], ], dtype=np.uint8)
        open_img = cv2.morphologyEx(img, op=cv2.MORPH_OPEN, kernel=kernel, anchor=(1, 1), iterations=1)
        self.de_img = open_img
        
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, self.de_img)
        
        pass
    
    def recurse_fill(self, img, seed):
        v, h = seed
        if img[v, h] != 128:
            return
        top_bias = self.walker_width if (v - self.walker_width > 0) else v
        bottom_bias = self.walker_width if (v + self.walker_width < img.shape[0]) else img.shape[0] - v
        left_bias = self.walker_width if (h - self.walker_width > 0) else h
        right_bias = self.walker_width if (h + self.walker_width < img.shape[1]) else img.shape[1] - h
        local_img = img[v - top_bias:v + bottom_bias, h - left_bias:h + right_bias]
        aims = np.array(np.where(local_img == 128)).T
        save_255 = np.array(np.where(local_img == 255)).T
        for aim in aims:
            cv2.line(local_img, (left_bias, top_bias), (aim[1], aim[0]), 128, )
        for i in save_255:
            local_img[i[0], i[1]] = 255
        img[v, h] = 255
        dircs = np.array([i for i in itertools.product([-1, 0, 1], [-1, 0, 1], )])
        for dirc in dircs:
            v_next, h_next = v + dirc[0], h + dirc[1]
            if ((v_next >= 0) and (v_next < img.shape[0])) and (
                    (h_next >= 0) and (h_next < img.shape[1])):
                self.recurse_fill(img=img, seed=(v_next, h_next))
    
    def fill_gap_img(self, src_img=None, seed_img=None, opath=None):
        src_img, seed_img = deepcopy(src_img), deepcopy(seed_img)
        src_img[src_img == 255] = 128
        seeds = np.array(np.where(seed_img == 255)).T
        
        # for seed in seeds:
        #     self.recurse_fill(img=src_img, seed=seed)
        #     self.fill_img = src_img
        #     if opath is not None:
        #         os.makedirs(os.path.dirname(opath), exist_ok=True)
        #         cv2.imwrite(opath, self.fill_img)
        #
        #     self.show_all()
        #
        stack = list(seeds)
        while len(stack) > 0:
            v, h = stack.pop()
            if src_img[v, h] != 128:
                continue
            self.count += 1
            print(self.count)
            top_bias = self.walker_width if (v - self.walker_width > 0) else v
            bottom_bias = self.walker_width if (v + self.walker_width < src_img.shape[0]) else src_img.shape[0] - v
            left_bias = self.walker_width if (h - self.walker_width > 0) else h
            right_bias = self.walker_width if (h + self.walker_width < src_img.shape[1]) else src_img.shape[1] - h
            local_img = src_img[v - top_bias:v + bottom_bias, h - left_bias:h + right_bias]
            aims = np.array(np.where(local_img == 128)).T
            save_255 = np.array(np.where(local_img == 255)).T
            for aim in aims:
                cv2.line(local_img, (left_bias, top_bias), (aim[1], aim[0]), 128, )
            for i in save_255:
                local_img[i[0], i[1]] = 255
            src_img[v, h] = 255
            dircs = np.array([i for i in itertools.product([-1, 0, 1], [-1, 0, 1], )])
            for dirc in dircs:
                v_next, h_next = v + dirc[0], h + dirc[1]
                if ((v_next >= 0) and (v_next < src_img.shape[0])) and (
                        (h_next >= 0) and (h_next < src_img.shape[1])):
                    stack.append((v_next, h_next))
        self.fill_img = src_img
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, self.fill_img)
    
    def skeleton_img(self, img=None, opath=None):
        bin_img = deepcopy(img)
        bin_img[bin_img == 255] = 1
        # skeleton = morphology.medial_axis(bin_img, return_distance=False)
        skeleton = morphology.skeletonize(bin_img)
        skeleton = np.array(skeleton, dtype=np.uint8) * 255
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0], ], dtype=np.uint8)
        skeleton = cv2.morphologyEx(skeleton, op=cv2.MORPH_DILATE, kernel=kernel, anchor=(1, 1), iterations=1)
        # skeleton = cv2.resize(skeleton, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        # skeleton = cv2.resize(skeleton, dsize=bin_img.T.shape, interpolation=cv2.INTER_CUBIC)
        # skeleton = cv2.resize(skeleton, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        # skeleton = cv2.resize(skeleton, dsize=bin_img.T.shape, interpolation=cv2.INTER_CUBIC)
        self.sk_img = skeleton
        
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, self.sk_img)
    
    def extract_free_space(self, img=None, opath=None):
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, img)
    
    def plot_lines_to_img(self, lines, img, opath=None):
        lines = np.array(lines, dtype=int)
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(img, (x1, y1), (x2, y2), 255, thickness=1)
        
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, img)
        
        return img
    
    def extract_lines(self, img=None, opath=None):
        # cv2.HoughLines(img, rho, theta, threshold, lines=None, srn=None, stn=None, min_theta=None, max_theta=None)
        # lines = cv2.HoughLinesP(img, rho=1, theta=np.pi / 180, threshold=3, lines=None, minLineLength=4,
        #                         maxLineGap=4)[:, 0, :]
        # lines = cv2.ximgproc.createFastLineDetector().detect(img)[:, 0, :]
        lines = cv2.HoughLinesP(img, rho=10, theta=np.pi / 45, threshold=200, lines=None, minLineLength=None,
                                maxLineGap=None)[:, 0, :]
        line_img = np.zeros_like(img)
        print('Number of extracted lines: ', len(lines))
        line_img = self.plot_lines_to_img(lines, line_img)
        
        self.line_img = line_img
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, self.line_img)
        
        return img, lines
    
    def calibrate_coordinate(self, img=None, opath=None):
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, img)
    
    def model_map(self, img=None, opath=None):
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, img)
    
    def segment_map(self, img=None, opath=None):
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, img)
    
    def convert_2_graph(self, img=None, opath=None):
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, img)
    
    def calculate_degree(self, lines):
        x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3],
        radian = np.pi / 2 if np.allclose(x2, x1, axis=0) else np.arctan((y2 - y1) / (x2 - x1))
        degree = radian / np.pi * 180.
        return degree
    
    def group_lines(self, lines, ):
        degrees = self.calculate_degree(lines)[:, np.newaxis]
        line_degrees = zip(lines, degrees, )
        ini_groups = [line_degrees]
        res_groups = []
        
        while len(ini_groups) > 0:
            group = ini_groups.pop()
            line, degree, = zip(*group)
            if max(degree) - min(degree) < 5:
                res_groups.append(list(zip(line, degree)))
                continue
            
            clf = KMeans(n_clusters=2)
            clf.fit(degree)
            # centers = clf.cluster_centers_  # 数据中心点
            clf_label = clf.labels_
            for i in range(2):
                idx = np.where(clf_label == i)
                # if len(idx) > 0:
                ini_groups.append(zip(np.array(line)[idx], np.array(degree)[idx], ))
        print('Number of groups: ', len(res_groups))
        for group in res_groups:
            line, degree = list(zip(*group))
            print(max(degree) - min(degree))
        
        # for i in range(len(res_groups)):
        #     line, degree = list(zip(*(res_groups[i])))
        #     for j in degree:
        #         plt.scatter(j[0], j[0], c=['r', 'g', 'b', ][i % 3])
        #     plt.scatter(np.mean(degree), np.mean(degree), marker='*', s=100)
        # plt.show()
        return res_groups
    
    def calculate_lengths(self, lines):
        lines = np.array(lines)
        point1, point2 = lines[:, :2], lines[:, 2:]
        lengths = np.linalg.norm(point1 - point2, ord=2, axis=-1)
        
        return lengths
    
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
    
    def calculate_general_line_form(self, line, ):
        x1, y1, x2, y2 = line
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        return a, b, c
    
    def calculate_lines_cross_point(self, line1, line2, exist_parallel=True):
        '''
        
        :param line1:
        :param line2:
        :param exist_parallel: False: Force to return the intersection point at infinity
        :return:
        '''
        a1, b1, c1 = self.calculate_general_line_form(line1)
        a2, b2, c2 = self.calculate_general_line_form(line2)
        D = a1 * b2 - a2 * b1
        if abs(D) < self.eps:
            if exist_parallel:
                return None
            else:
                D = np.sign(D) * (abs(D) + self.eps) + self.eps
        
        x = (b1 * c2 - b2 * c1) / D
        y = (a2 * c1 - a1 * c2) / D
        return (x, y)
    
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
        
        ac = (ap @ ab) * ab / (ab @ ab + self.eps)
        cp = ap - ac
        
        return np.array((a1 + ac[0], a2 + ac[1])), cp
    
    def merge_point_line(self, point, line):
        # merge_point_line
        #                 p
        #                 |
        #                 |
        # a---------b ----c
        p1, p2 = point
        a1, a2, b1, b2 = line
        ab = np.array([b1 - a1, b2 - a2])
        ap = np.array([p1 - a1, p2 - a2])
        
        gamma = (ap @ ab) / (ab @ ab + self.eps)
        ac = gamma * ab
        c1, c2 = a1 + ac[0], a2 + ac[1]
        
        if gamma <= 0:
            return np.concatenate(((c1, c2), (b1, b2)))
        elif gamma >= 1:
            return np.concatenate(((a1, a2), (c1, c2)))
        else:
            return line
    
    def calculate_distance2_point_2_line(self, point, line, details=False):
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
        
        gamma = (ap @ ab) / (ab @ ab + self.eps)
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
        else:
            return d_shortest
    
    def rotate_point(self, center, point, angle):
        cx, cy, px, py = *center, *point
        x = (px - cx) * np.cos(angle) - (py - cy) * np.sin(angle) + cx
        y = (py - cy) * np.cos(angle) + (px - cx) * np.sin(angle) + cy
        return (x, y)
    
    def rotate_line(self, center, line, angle):
        a = self.rotate_point(center, line[:2], angle)
        b = self.rotate_point(center, line[2:], angle)
        return np.concatenate((a, b))
    
    def calculate_lines_angle(self, line1, line2):
        ab1 = np.array(line1[:2]) - np.array(line1[2:])
        ab2 = np.array(line2[:2]) - np.array(line2[2:])
        cos_alpha = (ab1 @ ab2) / (np.linalg.norm(ab1, ord=2) * np.linalg.norm(ab2, ord=2))
        alpha = np.arccos(cos_alpha)
        # alpha = np.arccos(np.sign(cos_alpha) * (abs(cos_alpha) - self.eps))
        
        return alpha
    
    # def rotate_to_calibrate_line(self, line1, line2, line):
    #     # line1: a1 ------------- b1
    #     # line2:          a2 --------- b2    d_as: shortest distance from line2_a to line1
    #     # line : a ------------------- b
    #     length1, length2 = self.calculate_lengths([line1])[0], self.calculate_lengths([line2])[0]
    #     if length1 < length2:  # 长线在前
    #         line1, line2 = line2, line1
    #         length1, length2 = length2, length1
    #     ratio = length2 / (length1 + length2)
    #     cross_point = self.calculate_lines_cross_point(line1, line2, exist_parallel=False)
    #
    #     prj_points = []
    #     if cross_point is None:  # 线段平行
    #         _, p_vector = self.project_point_2_line(point=line1[:2], line=line2)
    #         p_vector *= ratio
    #         prj_points = [line1[:2] - p_vector, line1[2:] - p_vector, line2[:2] + p_vector, line2[2:] + p_vector]
    #     else:  # 线段相交
    #         alpha = self.calculate_lines_angle(line1, line2)
    #         alpha = np.pi / 2 - np.abs(np.pi / 2 - alpha)
    #         angel = alpha * ratio
    #         print('Degree contract:', alpha * 180 / np.pi, angel * 180 / np.pi)
    #
    #
    #         line_1 = self.rotate_line(cross_point, line, angel)
    #         line_2 = self.rotate_line(cross_point, line, -angel)
    #         alpha_1 = self.calculate_lines_angle(line_1, line2)
    #         alpha_1 = np.pi / 2 - np.abs(np.pi / 2 - alpha_1)
    #         alpha_2 = self.calculate_lines_angle(line_2, line2)
    #         alpha_2 = np.pi / 2 - np.abs(np.pi / 2 - alpha_2)
    #
    #         if alpha_1 <= alpha_2:
    #             return line_1
    #         else:
    #             return line_2
    #
    def merge_lines_per_pair(self, line1, line2):
        distance = self.calculate_distance2_per_line_pair(line1, line2)
        if distance <= self.nearest_distance2_per_line_pair:
            ### combine
            # calculate Projection line
            # line1: a1 ------------- b1
            # line2:          a2 --------- b2    d_as: shortest distance from line2_a to line1
            # line : a ------------------- b
            
            length1, length2 = self.calculate_lengths([line1])[0], self.calculate_lengths([line2])[0]
            if length1 < length2:  # 长线在前
                line1, line2 = line2, line1
                length1, length2 = length2, length1
            ratio = length2 / (length1 + length2)
            cross_point = self.calculate_lines_cross_point(line1, line2, exist_parallel=True)
            
            prj_points = []
            alpha = self.calculate_lines_angle(line1, line2)
            if (cross_point is None) or (alpha < np.pi / 180):  # 线段平行
                _, p_vector = self.project_point_2_line(point=line1[:2], line=line2)
                p_vector *= ratio
                prj_points = [line1[:2] - p_vector, line1[2:] - p_vector, line2[:2] + p_vector, line2[2:] + p_vector]
            else:  # 线段相交
                print('cross_point: {:.2f} ---- {:.2f}'.format(*cross_point))
                alpha = np.pi / 2 - np.abs(np.pi / 2 - alpha)
                angel = alpha * ratio
                print('Degree contract: {:.2f} ---- {:.2f}'.format(alpha * 180 / np.pi, angel * 180 / np.pi))
                
                line1_1 = self.rotate_line(cross_point, line1, angel)
                line1_2 = self.rotate_line(cross_point, line1, -angel)
                alpha_1 = self.calculate_lines_angle(line1_1, line2)
                alpha_1 = np.pi / 2 - np.abs(np.pi / 2 - alpha_1)
                alpha_2 = self.calculate_lines_angle(line1_2, line2)
                alpha_2 = np.pi / 2 - np.abs(np.pi / 2 - alpha_2)
                line = line1_1 if (alpha_1 <= alpha_2) else line1_2
                prj_point11, _ = self.project_point_2_line(point=line1[:2], line=line)
                prj_point12, _ = self.project_point_2_line(point=line1[2:], line=line)
                prj_point21, _ = self.project_point_2_line(point=line2[:2], line=line)
                prj_point22, _ = self.project_point_2_line(point=line2[2:], line=line)
                prj_points = [prj_point11, prj_point12, prj_point21, prj_point22]
            line_ls = list(itertools.combinations(prj_points, r=2))
            line_ls = [np.concatenate(i) for i in line_ls]
            idx = np.argmax(self.calculate_lengths(line_ls))
            
            return line_ls[idx]
            
            # alpha = self.calculate_lines_angle(line1, line2)
            # alpha = np.pi / 2 - np.abs(np.pi / 2 - alpha)
            # angel = alpha * length2 / (length1 + length2)
            # print('Degree contract:', alpha * 180 / np.pi, angel * 180 / np.pi)
            # cross_point = self.calculate_lines_cross_point(line1, line2, exist_parallel=True)
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
    
    def clean_lines(self, lines, min_length=3):
        lengths = self.calculate_lengths(lines)
        return np.delete(lines, obj=np.where(lengths < min_length)[0], axis=0)
    
    def merge_lines(self, lines):
        lines = self.clean_lines(lines)
        line_groups = self.group_lines(lines)
        
        combined_groups = []
        for group in line_groups:
            group_lines = self.combine_lines_per_group(group=group)
            combined_groups.append(group_lines)
        return combined_groups
    
    def find_contours(self, img=None, opath=None):
        contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE, )
        # mode: RETR_EXTERNAL RETR_LIST RETR_CCOMP RETR_TREE
        # method: CHAIN_APPROX_NONE CHAIN_APPROX_SIMPLE CHAIN_APPROX_TC89_L1 CHAIN_APPROX_TC89_KCOS
        self.test_img = np.zeros_like(img)
        # self.connect_contours(self.test_img, contours, )
        
        # for cnt in contours:
        # get the bounding rect
        # x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(self.test_img, (x, y), (x + w, y + h), 255, thickness=1)
        
        approx_contours = []
        color = list(range(100, 255, 10))
        print('Number of contours: ', len(contours))
        for i in range(len(contours)):
            cnt = contours[i]
            approx = cv2.approxPolyDP(cnt, epsilon=5, closed=False)
            approx_contours.append(approx)
            cv2.drawContours(self.test_img, [approx_contours[-1]], -1, color[i % 15], )
        # cv2.drawContours(self.test_img, approx_contours, -1, 255, )
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, self.test_img)
        return contours
    
    def connect_contours(self, img=None, contours=None, opath=None):
        cv2.drawContours(img, contours, -1, 100, )
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, self.test_img)
        return img
    
    def DBSCAN_cluster(self, img=None, opath=None):
        points = np.array(np.where(img == 255)).T
        clf = DBSCAN(eps=2 ** 0.5 + self.eps, min_samples=7, metric='euclidean', )
        ### 8邻域内至少有6个点才算是核心点 （7个点包括点本身）
        y_pred = clf.fit_predict(points)
        
        res_img = np.zeros_like(img)
        seg_len = []
        for i in set(y_pred):
            idx = np.array(np.where(y_pred == i))[0]
            print(i, len(idx))
            seg_len.append(len(idx))
        seg_len = np.array(seg_len)
        max_idx = np.argmax(seg_len)
        
        temp_img = np.zeros_like(img)
        idx = np.array(np.where(y_pred == max_idx))[0]
        cls_points = points[idx]
        for j in cls_points:
            temp_img[j[0]][j[1]] = 255
        print(temp_img)
        
        for i in set(y_pred):
            temp_img = np.zeros_like(img)
            idx = np.array(np.where(y_pred == i))[0]
            cls_points = points[idx]
            for j in cls_points:
                temp_img[j[0]][j[1]] = 255
            # approx = cv2.approxPolyDP(temp_img, epsilon=3, closed=False)
        
        self.model_img = res_img
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, self.model_img)
        return res_img
    
    def detect_corners(self, img=None, opath=None):
        res_img = deepcopy(img)
        # siftDetector = cv2.SIFT_create()
        # kps = siftDetector.detect(res_img, )
        kps = cv2.goodFeaturesToTrack(image=res_img, maxCorners=100, qualityLevel=0.000001, minDistance=3, blockSize=9)
        # src 单通道输入图像，八位或者浮点数。
        # maxCorners 表示最大返回关键点数目。
        # qualityLevel 表示拒绝的关键点 R < qualityLevel × max response将会被直接丢弃。
        # minDistance 表示两个关键点之间的最短距离。
        # mask 表示mask区域，如果有表明只对mask区域做计算。
        # blockSize 计算梯度与微分的窗口区域。
        # useHarrisDetector 表示是否使用harris角点检测，默认是false，为shi - tomas。
        # k = 0.04 默认值，当useHarrisDetector为ture时候起作用。
        res_img = res_img // 5
        for kp in kps:
            kp = list(map(int, kp[0]))
            cv2.circle(res_img, center=kp, radius=1, color=255, thickness=1)
        
        self.graph_img = res_img
        if opath is not None:
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            cv2.imwrite(opath, self.graph_img)
        return res_img
        
        # im = cv2.drawKeypoints(img_RGB, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    def __call__(self, ):
        print('Calling...')
        # 二值化
        self.binary_img(img=self.ini_img, opath=self.binary_path)
        # self.test_img = self.canny_img(img=self.ini_img, opath=self.test_path)
        # 膨胀腐蚀去噪点
        self.denoise_img(img=self.bi_img, opath=self.denoise_path)
        # 骨架提取？
        self.skeleton_img(img=self.de_img, opath=self.skeleton_path)
        # 提取轮廓
        contours = self.find_contours(img=self.sk_img, opath=self.test_path)
        self.DBSCAN_cluster(img=self.sk_img, opath=self.model_map_path)
        # 提取关键点
        self.detect_corners(img=self.sk_img, opath=self.graph_path)
        
        # 提取可入空间
        
        # 直线提取？
        _, lines = self.extract_lines(img=self.bi_img, opath=self.line_path)
        # 重分类，并合并直线
        # line_groups = self.merge_lines(lines)
        # lines = np.concatenate(line_groups, axis=0)
        # self.graph_img = np.zeros_like(self.sk_img)
        # self.plot_lines_to_img(lines=lines, img=self.graph_img, opath=self.graph_path)
        
        # 填充小缺口
        # self.fill_gap_img(src_img=self.sk_img, seed_img=self.sk_img, opath=self.fill_gap_path)
        
        # SLAM地图矫正
        
        # 选取坐标点，建模地图
        
        # 区域分割
        
        # 建图
        
        # sk_img = skeleton_img(bi_img)
        
        pass


if __name__ == '__main__':
    
    ini_img_dir = '../image/ini'
    ini_img_path_ls = [os.path.join(ini_img_dir, i) for i in os.listdir(ini_img_dir)]
    odir = '../image'
    
    for ini_img_path in ini_img_path_ls:
        print('\nProcessing %s ...' % ini_img_path)
        _map = MAP(ini_img_path=ini_img_path, odir=odir)
        _map()
        _map.show_all()
        print('Finish processing %s...\n' % ini_img_path)
        break
