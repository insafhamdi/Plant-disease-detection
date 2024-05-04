# Plant Leaf Disease Detection 

This project leverages a 7-layer (2 Convolutional Layers, 2 Pooling Layers, 3 Fully Connected Layers) deep convolutional neural network (CNN) to identify various plant leaf diseases from images. This is a significant contribution to agricultural research and practical applications in crop management.

## Project Contents

1. **Dataset**: A dataset containing 61,486 images of plant leaves, both diseased and healthy.
2. **Deep Learning Model**: Two 7-layer CNN models developed to classify the images into 39 distinct leaf health categories.
3. **Training and Evaluation Scripts**: Python scripts used for training the models and evaluating their performance.
4. **Documentation**: This README and additional documents that help understand and use the project.

## Dataset Details

The dataset includes images augmented using various techniques to enhance the model's robustness:

- Flipping
- Gamma correction
- Noise injection
- PCA color augmentation
- Rotation
- Scaling

### Dataset Classes

Images are classified into 39 categories, including specific diseases and healthy leaves.
1. Apple_scab
2. Apple_black_rot
3. Apple_cedar_apple_rust
4. Apple_healthy
5. Background_without_leaves
6. Blueberry_healthy
7. Cherry_powdery_mildew
8. Cherry_healthy
9. Corn_gray_leaf_spot
10. Corn_common_rust
11. Corn_northern_leaf_blight
12. Corn_healthy
13. Grape_black_rot
14. Grape_black_measles
15. Grape_leaf_blight
16. Grape_healthy
17. Orange_huanglongbing
18. Peach_bacterial_spot
19. Peach_healthy
20. Pepper_bacterial_spot
21. Pepper_healthy
22. Potato_early_blight
23. Potato_healthy
24. Potato_late_blight
25. Raspberry_healthy
26. Soybean_healthy
27. Squash_powdery_mildew
28. Strawberry_healthy
29. Strawberry_leaf_scorch
30. Tomato_bacterial_spot
31. Tomato_early_blight
32. Tomato_healthy
33. Tomato_late_blight
34. Tomato_leaf_mold
35. Tomato_septoria_leaf_spot
36. Tomato_spider_mites_two-spotted_spider_mite
37. Tomato_target_spot
38. Tomato_mosaic_virus
39. Tomato_yellow_leaf_curl_virus

## Deep Learning Model

The core of this project involves two 7-layer convolutional neural network models, designed to efficiently classify images into different categories of leaf diseases. The models were trained using PyTorch.

### Architecture

Details about the model architecture, including layer types, activation functions, etc.

### Training

Explain how the models were trained, including details such as learning rate, evaluation metrics, number of epochs, etc.
