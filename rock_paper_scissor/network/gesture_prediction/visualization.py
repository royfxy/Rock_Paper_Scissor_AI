import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def read_gesture_data(gesture):
    assert gesture in ["paper", "rock", "scissor"]
    data = np.load(
        "rock_paper_scissor/network/gesture_prediction/data/" + gesture + ".npy")
    data = (data - data.mean(axis=(1, 2, 3), keepdims=True)) / \
        data.std(axis=(1, 2, 3), keepdims=True)

    return data


def visualize_gesture_data(data):
    data = (data - data.mean()) / data.std()
    # normalize data to [0, 1]
    data = (data - data.min(axis=0, keepdims=True))
    data = data / np.max(data)
    data = np.swapaxes(data, 0, 1)
    data = -(data) + 1
    # display image
    x = ticker.MultipleLocator(1)
    y = ticker.MultipleLocator(1)
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(x)
    ax.yaxis.set_major_locator(y)
    # show data in gray scale
    ax.imshow(data, cmap="gray")
    plt.show()


def mean_gesture_data(data):
    # compute mean of each joint
    mean_data = np.mean(data, axis=0)
    return mean_data


if __name__ == "__main__":
    data = read_gesture_data("paper")
    data = mean_gesture_data(data)
    visualize_gesture_data(data)
