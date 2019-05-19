import numpy as np
from numpy import linalg as LA

def array2samples_distance(array1, array2):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1 
    """
    if array1[0][0]==665:
        num_point, num_features = array1.shape
        expanded_array1 = np.tile(array1, (num_point, 1))
        print("expanded_array1: "+ str(expanded_array1 ))
        expanded_array2 = np.reshape(
                np.tile(np.expand_dims(array2, 1), 
                        (1, num_point, 1)),
                (-1, num_features))
        print("expanded_array2: "+ str(expanded_array2 ))

        distances = LA.norm(expanded_array1-expanded_array2, axis=1)
        print("distances: " +str(distances))
        distances = np.reshape(distances, (num_point, num_point))
        print("distances: " +str(distances) )
        distances = np.min(distances, axis=1)
        print("distances: " +str(distances) )
        distances = np.mean(distances)
        print("distances: " +str(distances) )
        return distances

    else:
        num_point, num_features = array1.shape
        expanded_array1 = np.tile(array1, (num_point, 1))
        expanded_array2 = np.reshape(
                np.tile(np.expand_dims(array2, 1), 
                        (1, num_point, 1)),
                (-1, num_features))
        distances = LA.norm(expanded_array1-expanded_array2, axis=1)
        distances = np.reshape(distances, (num_point, num_point))
        distances = np.min(distances, axis=1)
        distances = np.mean(distances)
        return distances


def chamfer_distance_numpy(array1, array2):
    batch_size, num_point, num_features = array1.shape
    dist = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
     
        # print("av_dist1: " + str(av_dist1))
        # print("av_dist2: " + str(av_dist2))

        dist = dist + (av_dist1)/batch_size
    return dist


