import numpy as np
import pandas as pd
from vqa import clustering
from rus import *
from sklearn.decomposition import FastICA


# Placeholder for your PID calculation function
# Replace this with your actual PID calculation function
def decorrelate_inputs(input1, input2):
    """Decorrelates input1 and input2 using ICA."""
    X = np.stack((input1.flatten(), input2.flatten()), axis=1) #stack into a single array.
    ica = FastICA(n_components=2)
    X_ica = ica.fit_transform(X)
    return X_ica[:, 0].reshape(input1.shape), X_ica[:, 1].reshape(input2.shape)

def calculate_pid(input1, input2, output):
    """
    Placeholder for your PID calculation.
    Replace with your actual implementation.
    Returns dummy values for now.
    """
    kmeans_im, data_im = clustering(input1, pca=True, n_components=20, n_clusters=10)
    kmeans_txt, data_txt = clustering(input2, pca=True, n_components=20, n_clusters=10)
    kmeans_out, data_out = clustering(output, pca=True, n_components=20, n_clusters=10)

    kmeans_im, kmeans_txt, kmeans_out = kmeans_im.reshape(-1, 1), kmeans_txt.reshape(-1, 1), kmeans_out.reshape(-1, 1)
    P, maps = convert_data_to_distribution(kmeans_im, kmeans_txt, kmeans_out)
    res = get_measure(P)
    return res
                

def calculate_modality_dependence(res):
    redundancy, uniqueness1, uniqueness2, synergy = res['redundancy'], res['unique1'], res['unique2'], res['synergy']
    total = redundancy + uniqueness1 + uniqueness2 + synergy

    dependence1 = uniqueness1 / total
    dependence2 = uniqueness2 / total
    tot = dependence1 + dependence2
    dep1 = dependence1/tot
    dep2 = dependence2/tot

    return {
        "dependence1": dep1,
        "dependence2": dep2,
        "redundancy": redundancy,
        "unique1": uniqueness1,
        "unique2": uniqueness2,
        "synergy": synergy
    }

def generate_synthetic_data(num_sets=10, rows=50, cols=100):
    """Generates 25 sets of synthetic data as lists of 2D NumPy arrays."""
    all_sets = []
    for _ in range(num_sets):
        input1 = np.random.normal(loc=0, scale=1, size=(rows, cols))
        input2 = np.random.uniform(low=-0.5, high=0.5, size=(rows, cols))
        input1_decorrelated, input2_decorrelated = decorrelate_inputs(input1, input2)
        output = input2_decorrelated
        all_sets.append((input1_decorrelated, input2_decorrelated, output))
    return all_sets
import numpy as np

# def generate_synthetic_data(num_sets=10, rows=50, cols=100, threshold=0.1):
#     """Generates synthetic data with dot product thresholding."""
#     all_sets = []
#     for _ in range(num_sets):
        
#         input1 = np.random.normal(loc=0, scale=1, size=(rows, cols))
#         input2 = np.random.uniform(low=-0.5, high=0.5, size=(rows, cols))

#         # Flatten the arrays for dot product calculation
#         input1_flat = input1.flatten()
#         input2_flat = input2.flatten()

#         # Calculate dot product
#         dp = np.dot(input1_flat, input2_flat)

#         # Check if dot product is within threshold
#         if abs(dp) < threshold:
#             output = input1
#             all_sets.append((input1, input2, output))
#         else:
#             # If dot product exceeds threshold, regenerate inputs until it's within threshold
#             while abs(dp) >= threshold:
#                 input1 = np.random.normal(loc=0, scale=1, size=(rows, cols))
#                 input2 = np.random.normal(loc=0, scale=1, size=(rows, cols))
#                 input1_flat = input1.flatten()
#                 input2_flat = input2.flatten()
#                 dp = np.dot(input1_flat, input2_flat)
#             output = input1
#             all_sets.append((input1, input2, output))

#     return all_sets

def train_and_evaluate_model(data):
    """Prints the values from the NumPy arrays."""
    n = len(data)
    tot_1, tot_2, tot_3, tot_4 = 0, 0, 0, 0
    for i in range(n):
        # print(f"Input1: {data[i][0]}, Input2: {data[i][1]}, Output: {data[i][2]}")
        print("Iteration", i)
        res = calculate_pid(data[i][0], data[i][1], data[i][2])
        dep1, dep2 = calculate_modality_dependence(res)
        tot_1 += dep1
        tot_2 += dep2
        # tot_3 += un1
        # tot_4 += un2
        # print(dep1, dep2, bias)
        print()
    print("Bias 1:", tot_1/n)
    print("Bias 2:", tot_2/n)
    print()
    # print("Bias 1:", tot_3/n)
    # print("Bias 2:", tot_4/n)


        

# Main execution
synthetic_data = generate_synthetic_data()
# train_and_evaluate_model(synthetic_data)

# Calculate PID and dependence
# redundancy, uniqueness1, uniqueness2, synergy = calculate_pid(input1_test, input2_test, output_test)
# dependence1, dependence2, bias = calculate_modality_dependence(redundancy, uniqueness1, uniqueness2, synergy)

# print(f"Mean Squared Error: {mse}")
# print(f"Redundancy: {redundancy}")
# print(f"Uniqueness 1: {uniqueness1}")
# print(f"Uniqueness 2: {uniqueness2}")
# print(f"Synergy: {synergy}")
# print(f"Dependence Modality 1: {dependence1}")
# print(f"Dependence Modality 2: {dependence2}")
# print(f"Bias: {bias}")