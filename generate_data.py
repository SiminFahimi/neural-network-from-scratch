import numpy as np
import numpy as np

def generate_clean_3d_classification_data(count):
    x = np.random.uniform(-20, 20, (count, 1))
    y = np.random.uniform(-20, 20, (count, 1))

    surface = np.sin(x) + np.cos(y) 

    labels = np.random.randint(0, 2, (count, 1))
    
    offset = np.random.uniform(1.0, 3.0, (count, 1))

    z = np.where(
        labels == 1,
        surface + offset,
        surface - offset
    )

    x_data = np.hstack((x, y, z))
    y_data = labels

    return x_data, y_data


def data(count, data_eng=False):
    x_data, y_data = generate_clean_3d_classification_data(count)
    # for i in range(x_data.shape[1]):
    #     cor = pearson_correlation(x_data[:,i], y_data)

    #     if -0.5 < cor < 0.5:
    #         new_feature = x_data[:,i] ** 2
    #         x_data = np.column_stack((x_data, new_feature))


    # 80% train / 20% test
    split = int(0.8 * count)

    x_train = x_data[:split]
    y_train = y_data[:split]
    x_test = x_data[split:]
    y_test = y_data[split:]

    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)

    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std

    return x_train, y_train, x_test, y_test, x_mean, x_std

# def pearson_correlation(x, y):
#     x_mean= np.mean(x)
#     y_mean= np.mean(y)

#     r = (np.sum((x - x_mean) * (y - y_mean))) / (( np.sqrt(np.sum(np. square(x - x_mean)))) *  (np.sqrt(np.sum(np. square(y - y_mean)))))
#     return r

# def decision_tree(x):

#     pass