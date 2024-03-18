import torch
import numpy as np
#import torch.nn as nn
#from ultralytics import YOLO
#import cv2
# from .. import DEVICE # Imagine you have a GPU device and write code using it.

# class SimpleNN(nn.Module):
#     def __init__(self,N):       #N=M-1
#         super(SimpleNN,self).__init__()
#         self.bn1=nn.BatchNorm1d(N)
#         self.ff1 = nn.Linear(N,256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.relu=nn.ReLU()
#         self.ff2 = nn.Linear(256, 128)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.ff3 = nn.Linear(128, 2)
#         self.softmax=nn.Softmax()
#
#     def forward(self, x):
#         x=self.bn1(x)
#         x=self.ff1(x)
#         x=self.bn2(x)
#         x=self.relu(x)
#         x=self.ff2(x)
#         x=self.bn3(x)
#         x=self.relu(x)
#         x=self.ff3(x)
#         x=self.softmax(x)
#         return x

class FallDetection():
    def __init__(self):
        pass
        # self.yolo=YOLO('yolov8n-pose.pt')
        # self.nnet=SimpleNN(self.N)
        # self.nnet.load_state_dict(self.yolo.state_dict())
        # self.nnet.eval()

    def calculate_angles_between_two_vectors(self,vec1, vec2):  # vec1 and vec2 are numpy arrays
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return np.arccos(cos_angle) * 180 / np.pi

    def simple_moving_average(self,input_tensor):
        moving_avg_tensor = torch.zeros_like(input_tensor)
        cumulative_sum = 0.0

        for i in range(len(input_tensor)):
            cumulative_sum += input_tensor[i]
            moving_avg_tensor[i] = cumulative_sum / (i + 1)

        return moving_avg_tensor

    def sign1(self,skeleton_cache):
        pnt = (skeleton_cache[0][11] + skeleton_cache[0][12]) / 2
        pnt[1] = pnt[1] + torch.norm(skeleton_cache[0][6] - skeleton_cache[0][12]) / 4

        vector = torch.zeros(skeleton_cache.shape[0] - 1)
        for i in range(skeleton_cache.shape[0] - 1):
            # Derivative of the coordinate change of the person's center of gravity, which is
            pnt_next = (skeleton_cache[i + 1][11] + skeleton_cache[i + 1][12]) / 2
            pnt_next[1] = pnt_next[1] + torch.norm(skeleton_cache[i + 1][6] - skeleton_cache[i + 1][12]) / 4

            # First calculate the diff of y,and then in the call function *fps
            diff = torch.abs(pnt_next[1] - pnt[1])

            # Store the result in the vector
            vector[i] = diff

        return vector

    horizontal = np.array([100, 0])

    def sign2(self,skeleton_cache):
        angles = torch.zeros(skeleton_cache.shape[0] - 1)
        for i in range(skeleton_cache.shape[0] - 1):
            neck_center = (skeleton_cache[i][5] + skeleton_cache[i][6]) / 2
            hip_center = (skeleton_cache[i][11] + skeleton_cache[i][12]) / 2

            centerline = (neck_center - hip_center).numpy()  # (skeleton_cache[i][0] - the_center).numpy()
            angles[i] = self.calculate_angles_between_two_vectors(centerline, self.horizontal)

        return angles



    def __call__(self, skeleton_cache):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        '''
            This __call__ function takes a cache of skeletons as input, with a shape of (M x 17 x 2), where M represents the number of skeletons.
            The value of M is constant and represents time. For example, if you have a 7 fps stream and M is equal to 7 (M = 7), it means that the cache length is 1 second.
            The number 17 represents the count of points in each skeleton (as shown in skeleton.png), and 2 represents the (x, y) coordinates.

            This function uses the cache to detect falls.

            The function will return:
                - bool: isFall (True or False)
                - float: fallScore
        '''
        # Mx17x2->(M-1)

        #1 The person's center of gravity descent speed

        diff = self.sign1(skeleton_cache)
        diff = self.simple_moving_average(diff)

        # diff=torch.mul(diff,fps)      #for derivative

        mean_derivative_value = torch.mean(diff)
        # print(mean_derivative_value)

        threshold = 10

        is_fall = mean_derivative_value > threshold
        fall_score = mean_derivative_value / (threshold / 1.5 + mean_derivative_value)


        #2 the Angle between the Centerline of the Human and the Ground (horizontal vector)

        angles = self.sign2(skeleton_cache)
        angles = self.simple_moving_average(angles)
        mean_angle_value = torch.mean(angles)
        if mean_angle_value > 100:
            mean_angle_value = abs(180 - mean_angle_value)
        #print(mean_angle_value)

        if is_fall == True and mean_angle_value > 77:
            is_fall = False
            fall_score = (torch.rand(1, dtype=torch.float32) * 0.1 + 0.3).item()

        if is_fall == False and mean_angle_value < 45:
            is_fall = True
            fall_score = (torch.rand(1, dtype=torch.float32) * 0.1 + 0.8).item()

        return is_fall, fall_score
