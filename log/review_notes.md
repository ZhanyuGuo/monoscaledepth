# Review Notes

## Introduction

- **Monocular depth estimation** from Red-Green-Blue (RGB) images is a well-studied ill-posed problem in computer vision.

- Estimating depth from two-dimensional images plays an important role in various **applications** including **scene reconstruction**, **3D object detection**, **medical imaging**, **robotics** and **autonomous driving**.

- The **classical** depth estimation approaches rely on **multi-view geometry**, such as stereo image. **Multi-view** methods acquire depth information by utilizing visual cues and different camera parameters. However, their **computational time** and **memory requirements** are  important challenges for many applications. **Monocular** methods can solve the **memory requirement** issue, but it is **computationally difficult** to capture the global properties of a scene such as texture variation or defocus information.

- The advancement of **Convolutional Neural Networks (CNN)** and publicly available datasets have significantly improved the performance of monocular depth estimation methods.


## An Overview of Monocular Depth Estimation

- The concept of **depth estimation** refers to the process of preserving 3D information of the scene using 2D information captured by cameras. **Monocular** solutions use only one image. There are a variety of devices commercially available to provide depth information, however their **processing power**, **computational time**, **range limitation** and **cost** make them impractical for consumer devices.

- Sensors such as Kinect are categorized as **Time-of-Flight (ToF)** where the depth information is acquired by calculating the time required for a ray of light to travel from a light source to an object and back to the sensor. ToF sensors are more suitable for the **indoor environment** and **laser-based scanners (LiDAR)** are utilized for 3D measurement in the **outdoor environment**. LiDAR sensors have **high resolution**, **accuracy**, **performance in low light** and **speed**, but they are **expensive** and require **extensive power resources**.

- **Monocular depth estimation** methods perform with a **relatively small number of operations** and **in less computation time**. They **do not require alignment** and **calibration** which is important for multi-camera, or multi-sensor depth measurement systems. They play important roles in **understanding 3D scene geometry** and **3D reconstruction**, particularly in **cost-sensitive** applications and use cases.

### Problem Representation

- Let $I \in \mathbb{R}^{w \times h}$ image with size $w \times h$. The goal is to estimate the corresponding depth information $D\in \mathbb{R}^{w \times h}$. This is an ill-posed problem as there is an ambiguity in the scale of the depth. 

### Traditional Methods for Depth Estimation

- Traditional methods can be categorized in two sets, active and passive methods.

#### Active Methods

- Active methods involve computing the depth in the scene by interacting with the objects and the environment. **Light-based depth estimation**, **ultrasound** and **ToF**. These methods use the known speed of the wave to measure the time an emitted pulse takes to arrive at an image sensor.

#### Passive Methods

- Passive methods exploit the optical features of captured images. These methods involve extracting the depth information by computational image processing. Two approaches: **multi-view** depth estimation and **monocular** depth estimation. 

- The **traditional** depth estimation methods have various **limitations** including **computational complexity** and **associated high energy requirements**. **Deep learning** methods achieve **more accurate** results with **lower computational** and **energy demands**.

### Datasets

- NYU-v2, Make3D, KITTI, Pandora, SceneFlow.

## Deep Learning and Monocular Depth Estimation

- The majority of the deep learning-based methods involve a CNN trained on RGB-images and the corresponding depth maps. These methods can be categorized into **supervised**, **semi-supervised** and **self-supervised**.  

(to be continued)
