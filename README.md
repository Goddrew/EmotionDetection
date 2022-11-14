# Emotion Detection

## About

This project aims to detect facial expression in real time. The CNN model I created is an re-implementation of the papers below with an added `AdaptiveAvgPool2d` layer and `LazyLinear` layer. The CNN is trained on [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013), with 7 expressions including: **Angry, Disgust, Fear, Happy, Sad, Suprise, Neutral**. The model is built using PyTorch. OpenCV is used for video processing and Flask is used to build the rest of the web interface. 

## Training Process & Result

I use Adam as my optimization algorithm. The learning rate which gave me the best performance is `0.001` with a weight decay of `1e-6`. The model is trained with 50 epochs. It has an accuracy of around 53% on the testing data. However, the performance is quite well when it is used in the web application. 

## Future Work 

Even though the application works quite well, the model's performance on the testing set is not as well as I hoped. The State of the Art performance is around 75%. For future's work, I will look at the difference between the SOTA model and mine and see how they differ and take inspirations from it to improve my model's accuracy. I will be taking suggestions from [this](https://arxiv.org/pdf/1612.02903.pdf) paper and potentially others. 

## Acknowledgments

Current model inspired from

1. [Facial Expression Recognition with Deep Learning](http://cs230.stanford.edu/projects_winter_2020/reports/32610274.pdf)
2. [Emotion detection using deep learning](https://github.com/atulapra/Emotion-detection)

