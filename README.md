# LipReading

This is the keras implementation of *Lip2AudSpec: Speech reconstruction from silent lip movements video.* 

![Main Network](figures/Network_main.png)

### Abstract
In this study, we propose a deep neural network for reconstructing intelligible speech from silent lip movement videos. We use auditory spectrogram and its corresponding sound generation method which preserves pitch information resulting in a more natural sounding reconstructed speech. Our proposed network consists of an autoencoder to extract bottleneck features from the auditory spectrogram which is then used as target to our main lip reading network comprising of CNN, LSTM and fully connected layers. Our experiments show that the autoencoder is able to reconstruct the original auditory spectrogram with a 98% correlation and also improves the quality of reconstructed speech from the main lip reading network.
Our model, trained jointly on different speakers is able to extract individual speaker characteristics and gives promising results of reconstructing intelligible speech with superior word recognition accuracy.

### Demo

You can find all demo files [here](demo/).

A few samples of the network output are given below:

**Sample 1**

[![Sample1](https://img.youtube.com/vi/Op7Z9KH5Fis/0.jpg)](https://youtu.be/Op7Z9KH5Fis "Sample1_s1")

**Sample 2**

[![Sample2](https://img.youtube.com/vi/O0Gfb-1lu2k/0.jpg)](https://youtu.be/O0Gfb-1lu2k "Sample2_s29")



