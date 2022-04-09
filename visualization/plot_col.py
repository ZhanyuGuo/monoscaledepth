import numpy as np
import matplotlib.pyplot as plt


def main():
    depth = np.load("./10902.npy")
    depth = depth.squeeze()
    a = 1 / depth[:, 65]
    plt.plot(a)
    plt.show()


if __name__ == "__main__":
    main()
