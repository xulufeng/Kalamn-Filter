# 不调用库，自己写一个卡尔曼滤波器

import numpy as np
import matplotlib.pyplot as plt

def kalman_filter_function(position):
    # 输入参数是测量到的位置信息,[(x1, y1), (x2, y2), ...]

    #参数设定
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) #测量矩阵
    A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) #状态转移矩阵
    P_k_1 = np.eye(4) * 0.001  # 初始状态协方差矩阵
    Q = np.eye(4) * 0.001  # 噪声协方差矩阵
    R = np.zeros((2, 2))  # 测量噪声协方差矩阵

    predict_position = []
    X_k_1 = np.array([position[0][0], position[0][1], 0, 0])
    predict_position.append([X_k_1[0], X_k_1[1]]) #将测量的初值作为预测的初值，将其加入预测结果中

    #先对现有的测量值进行滤波
    for Z in position[1:]:
        Z_k = np.array([Z[0], Z[1]])
        #预测
        X_k_ = A @ X_k_1
        P_k_ = A @ P_k_1 @ A.T + Q

        #更新
        K_k= (P_k_ @ H.T) @ np.linalg.inv(H @ P_k_ @ H.T + R)
        X_k = X_k_ + K_k @ (Z_k - H @ X_k_)
        P_k = P_k_ - K_k @ H @ P_k_

        #迭代
        X_k_1 = X_k
        P_k_1 = P_k

        #保存滤波结果
        predict_position.append([X_k[0], X_k[1]])

    #再对未来的状态值进行预测
    for i in range(0, 3):
        X_k = A @ X_k_1
        X_k_1 = X_k
        predict_position.append([X_k[0], X_k[1]])

    return predict_position


if __name__ == "__main__":
    historical_position = [(4, 300), (61, 256), (116, 214), (170, 180), (225, 148), (279, 120), (332, 97),
         (383, 80), (434, 66), (484, 55), (535, 49), (586, 49), (634, 50),
         (683, 58), (731, 69), (778, 82), (824, 101), (870, 124), (917, 148),
         (962, 169), (1006, 212), (1051, 249), (1093, 290)]

    predict_position = kalman_filter_function(historical_position)

    history_trajectory = np.array(historical_position)
    predicted_trajectory = np.array(predict_position)

    plt.plot(history_trajectory[:, 0], history_trajectory[:, 1], marker='o', label='historical_position')
    plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], linestyle='dashed', marker='o', label='predict_position')
    plt.legend()
    plt.title('Kalman Filter Trajectory Prediction')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()
