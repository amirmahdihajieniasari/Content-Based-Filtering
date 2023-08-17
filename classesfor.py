from collections import defaultdict
import csv
import numpy as np
from numpy import genfromtxt
import pickle
import tabulate


def load_spec():
    # prepared datasets
    item_train = genfromtxt('./datasets/content_item_train.csv', delimiter=',')
    user_train = genfromtxt('./datasets/content_user_train.csv', delimiter=',')
    y_train = genfromtxt('./datasets/content_y_train.csv', delimiter=',')
    with open('./datasets/content_item_train_header.txt', newline='') as f:
        item_features = list(csv.reader(f))[0]
    with open('./datasets/content_user_train_header.txt', newline='') as f:
        user_features = list(csv.reader(f))[0]
    item_vecs = genfromtxt('./datasets/content_item_vecs.csv', delimiter=',')
    movie_dict = defaultdict(dict)
    count = 0
    with open('./datasets/content_movie_list.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in reader:
            if count == 0:
                count += 1  # skip header
            else:
                count += 1
                movie_id = int(line[0])
                movie_dict[movie_id]["title"] = line[1]
                movie_dict[movie_id]["genres"] = line[2]
    with open('./datasets/content_user_to_genre.pickle', 'rb') as f:
        user_to_genre = pickle.load(f)
    return item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre


def print_pred_movies(y_p, item, movie_dict, maxcount=10):
    count = 0
    disp = [["y_p", "movie id", "rating ave", "title", "genres"]]
    for i in range(0, y_p.shape[0]):
        if count == maxcount:
            break
        count += 1
        movie_id = item[i, 0].astype(int)
        disp.append(
            [np.around(y_p[i, 0].astype(float), 1), item[i, 0].astype(int), np.around(item[i, 2].astype(float),
                                                                                      1),
             movie_dict[movie_id]['title'], movie_dict[movie_id]['genres']])
    table = tabulate.tabulate(disp, tablefmt='grid', headers="firs trow")
    print(table)


def gen_user_vecs(user_vec, num_items):
    # given a user vector and convert it to user_vecs to match the size of item_vecs
    user_vecs = np.tile(user_vec, (num_items, 1))
    return user_vecs
