import time
import torch

start_time = time.time()

T1 = torch.eye(4)
T2 = torch.eye(4)

T1[:3, -1] = torch.tensor([1, 2, 3])
T2[:3, :3] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

T = torch.stack((T1, T2))

for _ in range(10000):
    # Method 1
    T = torch.inverse(T)

end_time = time.time()
print("Direct inverse: \n", T)
print("Duration: ", end_time - start_time)

start_time = time.time()
for _ in range(10000):
    # Method 2
    # T_clone = T.clone()

    # R = T_clone[:, :3, :3]
    R = T[:, :3, :3]
    R = R.transpose(1, 2)

    # t = T_clone[:, :3, 3:]
    t = T[:, :3, 3:]
    t = -torch.matmul(R, t)

    # T[:, :3, :3] = R
    # T[:, :3, 3:] = t

print("Computation:\n", T)

end_time = time.time()
print("Duration: ", end_time - start_time)
