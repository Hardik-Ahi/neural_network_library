import numpy as np
from numpy.random import default_rng
import pandas as pd

def get_vector(seed = 1, upper_bound = 10, n_samples = 10):
    generator = default_rng(seed)
    vector = upper_bound * generator.random((n_samples, 1))  # refer docs; [(upper - lower) * rng.random() + lower] gives range[lower, upper)
    return vector

def and_gate_dataset(positive_samples = 100, seed = 1000):
    generator = default_rng(seed)
    seeds = generator.choice(1000, 4, replace = False)
    # features
    both_positive = np.hstack((get_vector(seed = seeds[0], n_samples = positive_samples), get_vector(seed = seeds[1], n_samples = positive_samples)))
    one_zero_1 = np.hstack((np.zeros((positive_samples//2, 1)), get_vector(seed = seeds[2], n_samples = positive_samples//2)))
    one_zero_2 = np.hstack((get_vector(seed = seeds[3], n_samples = positive_samples//2), np.zeros((positive_samples//2, 1))))
    both_zero = np.array([0, 0])
    features = np.vstack((both_positive, one_zero_1, one_zero_2, both_zero))

    # labels
    labels_positive = np.ones((positive_samples, 1), dtype = int)
    labels_negative = np.zeros((positive_samples//2 + positive_samples//2 + 1, 1), dtype = int)
    labels = np.vstack((labels_positive, labels_negative))

    dataset = np.hstack((features, labels))
    generator.shuffle(dataset)
    return dataset[:, [0, 1]], dataset[:, [2]]

def get_minibatch(features, targets, batch_size = 1, start_at = 0):
    # give this the same stuff every time
    if start_at >= features.shape[0]:
        print("invalid start_at for get_minibatch")
        return None, None
    return features[start_at:min(features.shape[0], start_at + batch_size)], targets[start_at:min(targets.shape[0], start_at + batch_size)]

@np.vectorize(excluded = {1, 2, "mean", "std"})
def make_standard(x, mean, std):
    return (x - mean) / std

def standardize_data(dataframe, columns, from_means = None, from_stds = None):  # in-place operation
    for col in columns:
        dataframe[col] = dataframe[col].astype(float)
    means = dataframe[columns].mean() if from_means is None else from_means
    stds = dataframe[columns].std() if from_stds is None else from_stds
    dataframe.loc[:, columns] = (dataframe[columns] - means) / stds
    return means, stds
    

def linear_regression_dataset(samples = 200, x_start = 0, x_end = 100, slope = 1, intercept = 5, sigma_ = 8, seed = 7):
    x = np.linspace(x_start, x_end, samples)

    y = slope * x + intercept
    y += default_rng(seed).normal(0, sigma_, samples)

    return x.reshape((x.size, 1)), y.reshape((y.size, 1))

def equalize_classes(dataframe, target_name):
    if target_name not in dataframe.columns:
        print(f"invalid target name: {target_name}")
        return
    
    counts = dataframe[target_name].value_counts()
    least = counts.min()
    to_remove = counts - least
    
    for class_name in to_remove.index:
        temp = dataframe[dataframe[target_name] == class_name]
        drop_count = to_remove[class_name]
        if drop_count == 0:
            continue
        print(f'dropping {drop_count} for class: {class_name}')
        dataframe.drop(temp.iloc[:drop_count].index, inplace = True)

    dataframe.reset_index(drop = True, inplace = True)

def split_classes(dataframe, target_name, test_size = 0.2):
    if target_name not in dataframe.columns:
        print(f'invalid target name: {target_name}')
        return
    
    counts = pd.Series(dataframe[target_name].value_counts() * test_size, dtype = int)  # we need this many counts of each class in test set
    frames = []
    # simply transfer first 'n' rows for each class from main df to test df.
    for class_name in counts.index:
        temp = dataframe[dataframe[target_name] == class_name]
        copy_df = temp.iloc[:counts[class_name]]
        frames.append(copy_df)
        dataframe.drop(copy_df.index, inplace = True)
    
    train_df = dataframe.reset_index(drop = True)
    test_df = pd.concat(frames)
    test_df.reset_index(drop = True, inplace = True)
    return train_df, test_df

def split_data(dataframe, test_size = 0.2, seed = 1):
    test_df = dataframe.sample(frac = test_size, random_state = seed)
    train_df = dataframe.drop(test_df.index)
    train_df.reset_index(drop = True, inplace = True)
    test_df.reset_index(drop = True, inplace = True)

    return train_df, test_df

def pca(dataframe, target_name, n_components = 2):  # assume all features are valid (continuous numeric) for pca
    if target_name not in dataframe.columns:
        print(f'invalid target name: {target_name}')
        return
    features = dataframe.drop(columns = [target_name])

    # 1. standardize vars
    for col in features.columns:
        mean = features[col].mean()
        std = features[col].std()
        features[col] = (features[col] - mean)/std
    
    # 2. covariance matrix
    array = features.to_numpy()
    matrix = np.cov(array, rowvar = False)
    
    # 3. eigenvalues, eigenvectors
    evalues, evectors = np.linalg.eig(matrix)
    eframe = pd.DataFrame(np.hstack((evectors.T, evalues.reshape(evalues.size, 1))))  # rows = vectors
    eframe.sort_values(by = eframe.columns.size-1, ascending = False, inplace = True)
    eframe.drop(columns = [eframe.columns.size-1], inplace = True)
    
    # 4. form feature vector
    if n_components > eframe.shape[0]:
        print(f'asking too many components: {n_components} for {eframe.shape[0]} rows')
    vectors = eframe.iloc[:n_components].to_numpy().T

    # 5. recast data along feature vector
    result = np.dot(array, vectors)  # MxN data * NxP vectors = MxP result
    return result