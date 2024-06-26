[![White-Minimalist-Corporate-Personal-Profile-Linked-In-Banner.png](https://i.postimg.cc/Cxm7dKtF/White-Minimalist-Corporate-Personal-Profile-Linked-In-Banner.png)](https://github.com/shreyassn)


## About

The project, "Identification of Suspect in Crowd Using Face Recognition with Deep Learning," aims to develop an advanced system for accurately identifying suspects in crowded environments. Leveraging state-of-the-art face recognition and deep learning techniques, the system addresses challenges such as rapid detection, recognition under varying lighting conditions, non-frontal faces, and masked faces. This project was undertaken as our BTech final year project, with collaborative efforts among three team members under the guidance of a professor.

## Flow Diagram

[![Suspect-Detection-Methodology-drawio.png](https://i.postimg.cc/9MLM6vNF/Suspect-Detection-Methodology-drawio.png)](https://postimg.cc/CBfSD6gW)
## Prerequisites

- Numpy
- Pandas
- Matplotlib
- Scikit-Learn
- TensorFlow
- Keras
- Python
- OpenCV
- Dlib
- PyTorch
- YOLO

## Dataset Used




| Dataset Name | Purpose     | Size and Characteristics                |
| :-------- | :------- | :------------------------- |
| `WIDER Face` | `Face detection` | 32,203 annotated face images |
| `Dark Face` | `Face Detection & Recognition` | 6100 images under low light condition |
| `CelebA` | `Face Recognition` | 87,628 face images |
| `LFW Dataset` | `Evaluation` | 13,000 labeled face images |
| `Pascal Face` | `Evaluation & Testing` | 851 images |
| `UTK Face` | `Testing` | 20,000 face images |





## Data Augmentation

Data augmentation enriches datasets for better suspect detection in crowds by altering existing data. Techniques include geometric transformations (e.g., cropping, rotation) and photometric changes (e.g., brightness adjustment, noise addition), improving model accuracy and reliability (Wu et al., 2021).

[![Screenshot-2024-06-27-004726.png](https://i.postimg.cc/DyW5mFhs/Screenshot-2024-06-27-004726.png)](https://postimg.cc/PPj1FgPr)

Integration of Virtual Objects
Augmented Reality (AR) enhances facial recognition by introducing diverse facial appearances (e.g., skin tones, accessories). This reduces biases and improves accuracy across demographics, aiding algorithm optimization and real-world deployment (Mash et al., 2020).

Suspect Recognition
Using DeepFace, faces are detected and aligned using VGG-Face and Dlib. Facial features are represented as embeddings for accurate recognition, employing metrics like cosine similarity and Euclidean distance for measurement (DeepFace documentation).

[![Screenshot-2024-06-27-003618.png](https://i.postimg.cc/Jh3pmqL2/Screenshot-2024-06-27-003618.png)](https://postimg.cc/B86TB2tT)

[![Screenshot-2024-06-27-003637.png](https://i.postimg.cc/B6qMvKZQ/Screenshot-2024-06-27-003637.png)](https://postimg.cc/QByp4VWR)
## Suspected Identity Storage

The suspect database is built by collecting photos under varying lighting conditions and non-frontal angles. Each photo undergoes rigorous data augmentation to match real-world CCTV images, addressing challenges like partial face obstruction, masked faces, extreme angles (up to 180 degrees), varying lighting, makeup alterations, and added facial hair. These augmented images are then used for matching against faces extracted from CCTV footage.
## Face detection

We used YOLOv8, a state-of-the-art object detection tool, trained on a custom facial dataset for accurate detection of multiple faces per frame. Detected faces were precisely cropped and then passed to the facial recognition step. To avoid redundant cropping of the same face across video frames, we employed a nearest neighbor approach for efficient object tracking, ensuring seamless continuity.
## Face Recognition

For face recognition, we implemented FaceNet and DeepFace models due to their superior accuracy and rapid recognition capabilities, chosen after evaluating five different models. These models extract facial features from cropped images of detected faces, allowing precise identification. Integrated with a suspect database, the system quickly compares each face against known suspects, enabling efficient identification and real-time alerts to authorities.
## Prerequisites

- Numpy
- Pandas
- Matplotlib
- Scikit-Learn
- TensorFlow
- Keras
- Python
- OpenCV
- Dlib
- PyTorch
- YOLO

## References

- H. Verma, S. Lotia, and A. Singh, “Convolutional neural network based criminal detection,” in 2020 IEEE REGION 10 CONFERENCE (TENCON), pp. 1124–1129, 2020.
- S. T. Ratnaparkhi, A. Tandasi, and S. Saraswat, “Face detection and recognition for criminal identification system,” in 2021 11th International Conference on Cloud Computing, Data Science Engineering (Confluence), pp. 773–777, 2021.
- L. Zhang, J. Liu, B. Zhang, D. Zhang, and C. Zhu, “Deep cascade model-based face recognition: When deep-layered learning meets small data,” IEEE Transactions on Image Processing, vol. 29, pp. 1016–1029, 2020.
- Y. Feng, S. Yu, H. Peng, Y.-R. Li, and J. Zhang, “Detect faces efficiently: A survey and evaluations,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 4, no. 1, pp. 1–18, 2022.
- H. Wu, Z. Lu, J. Guo, and T. Ren, “Face detection and recognition in complex environments,” in 2021 40th Chinese Control Conference (CCC), pp. 7125–7130, 2021.
- J. Zhao, L. Xiong, J. Li, J. Xing, S. Yan, and J. Feng, “3d-aided dual-agent gans for unconstrained face recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 10, pp. 2380–2394, 2019.
- H. Qiu, D. Gong, Z. Li, W. Liu, and D. Tao, “End2end occluded face recognition by masking corrupted features,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 6939–6952, 2022.
- M. Wang and W. Deng, “Adaptive face recognition using adversarial information network,” IEEE Transactions on Image Processing, vol. 31, pp. 4909–4921, 2022.
- Y.-J. Ju, G.-H. Lee, J.-H. Hong, and S.-W. Lee, “Complete face recovery gan: Unsupervised joint face rotation and de-occlusion from a single-view image,” in 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 1173–1183, 2022.
- G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller, “Labeled faces in the wild: A database for studying face recognition in unconstrained environments,” Tech. Rep. 07-49, University of Massachusetts, Amherst, October 2007.
- E. Richardson, Y. Alaluf, O. Patashnik, Y. Nitzan, Y. Azar, S. Shapiro, and D. Cohen-Or, “Encoding in style: a stylegan encoder for image-to-image translation,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2287–2296, 2021.
- N. Y. Katkar and V. K. Garg, “Detection and tracking the criminal activity using network of cctv cameras,” in 2022 3rd International Conference on Smart Electronics and Communication (ICOSEC), pp. 664–668, 2022.
- J. Wang and Z. Xu, “Crowd anomaly detection for automated video surveillance,” in 6th International Conference on Imaging for Crime Prevention and Detection (ICDP-15), pp. 1–6, 2015.
- E. L. Andrade, R. B. Fisher, and S. Blunsden, “Detection of emergency events in crowded scenes,” in 2006 IET Conference on Crime and Security, pp. 528–533, 2006.
- M. K and L. Sujihelen, “Behavioural analysis for prospects in crowd emotion sensing: A survey,” in 2021 Third International Conference on Inventive Research in Computing Applications (ICIRCA), pp. 735–743, 2021.
- H. Idrees, I. Saleemi, C. Seibert, and M. Shah, “Multi-source multi-scale counting in extremely dense crowd images,” in 2013 IEEE Conference on Computer Vision and Pattern Recognition, pp. 2547–2554, 2013.
- K. K. Kumar and H. V. Reddy, “Literature survey on video surveillance crime activity recognition,” in 2022 First International Conference on Artificial Intelligence Trends and Pattern Recognition (ICAITPR), pp. 1–8, 2022.
- S. Singla and R. Chadha, “Detecting criminal activities from cctv by using object detection and machine learning algorithms,” in 2023 3rd International Conference on Intelligent Technologies (CONIT), pp. 1–6, 2023.
