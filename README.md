# Character Recognition


Currently: Developing a character recognition web app to allow users to draw on a canvas and have a CNN predict the character or digit
                           
Developed and Trained a Convolutional Neural Netork using the  [EMNIST Dataset](https://www.nist.gov/node/1298471/emnist-dataset). (Built using Keras)


- [x] Develop CNN architecture in Keras
  - Conv2D -> Conv2D -> MaxPooling2D -> Dropout -> Flatten -> Dense -> Dropout -> Dense
- [x] Train CNN 
  - 30 epochs (87% accuracy) 
- [x] Convert Keras Model (h5) to TensorflowJS (json/bin)
- [ ] Develop UI for Canvas Input