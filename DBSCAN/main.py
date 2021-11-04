'''
Created on Feb 13, 2014

@author: sushant
'''

import csv
import sys
import cv2
import numpy as np
from dbscanner import DBScanner

CONFIG = 'config'
DATA = 'data/abc.csv'


def get_data(config):
    data = []
    with open(DATA, 'rb') as file_obj:
        csv_reader = csv.reader(file_obj)
        for id_, row in enumerate(csv_reader):
            if len(row) < config['dim']:
                print("ERROR: The data you have provided has fewer \
                    dimensions than expected (dim = %d < %d)"
                      % (config['dim'], len(row)))
                sys.exit()
            else:
                point = {
                    'id'   : id_,
                    'value': []
                }
                for dim in range(0, config['dim']):
                    point['value'].append(float(row[dim]))
                data.append(point)
    return data


def main():
    config = {
        'eps'    : 2.5,
        'min_pts': 2,
        'dim'    : 2,
    }
    dbc = DBScanner(config)
    # data = get_data(config)
    data_point = [[0.697, 0.460],
                  [0.774, 0.376],
                  [0.634, 0.264],
                  [0.608, 0.318],
                  [0.556, 0.215],
                  [0.403, 0.237],
                  [0.481, 0.149],
                  [0.437, 0.211],
                  [0.666, 0.091],
                  [0.243, 0.267],
                  [0.245, 0.057],
                  [0.343, 0.099],
                  [0.639, 0.161],
                  [0.657, 0.198],
                  [0.360, 0.370],
                  [0.593, 0.042],
                  [0.719, 0.103],
                  [0.359, 0.188],
                  [0.339, 0.241],
                  [0.282, 0.257],
                  [0.748, 0.232],
                  [0.714, 0.346],
                  [0.483, 0.312],
                  [0.478, 0.437],
                  [0.525, 0.369],
                  [0.751, 0.489],
                  [0.532, 0.472],
                  [0.473, 0.376],
                  [0.725, 0.445],
                  [0.446, 0.459], ],
    img = cv2.imread('../image/denoise/0.jpg', cv2.IMREAD_GRAYSCALE)
    data_point = np.array(np.where(img == 255)).T
    
    data = []
    for i in range(len(data_point)):
        data.append({
            'id'   : i,
            'value': data_point[i]
        })
    
    dbc.dbscan(data)
    dbc.export()


if __name__ == "__main__":
    main()
