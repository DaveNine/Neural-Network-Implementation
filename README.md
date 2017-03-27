# Neural-Network-Implementation

This is my Neural Network Implementation, currently working on it to be able to handle the classic MNIST dataset. Working towards implementing
different cost functions / activation functions to see how it affects deep learning, as well as different strategies that help deep learning
learn faster, such as alternate learning algorithms other than Stochastic Gradient Descent (Conjugate-Gradient and Quasi-Newton Methods, as 
well as genetic algorithms).

Eventually, it is a goal to move towards convolutional neural networks since they seem to handle higher resolution images better, due to the number
of inputs in an image being lower, to handle the curse of dimensionality. I would also like to experiment with NEAT (genetic algorithms
in combination with varying the topology of a NN).

Further, I'd like to do something cool with it. Perhaps try to replicate TensorKart, https://github.com/kevinhughes27/TensorKart, in the form of Super mario or something slightly more basic and less computationally expensive. His approach is a good one -- rather than writing in LUA -- a common scripting language that can be utilized with many emulators, giving you full access to the emulated RAM as a mini-API in the emulator -- one can use screenshots of the current frame as inputs to the NN. Unfortunately this means we will have to play through a level to "teach" super mario how to run through it. After a while (and maybe enough training?), super mario's NN *should* be good enough to handle a level on it's own, maybe with a bit of reinforced training as mentioned in Kevin's implementation.  My plan is as follows:

~~~~~~~~~~~PHASE I~~~~~~~~~~~

1. Finish NN deep learning tutorial w/ MNIST

2. Figure out whether to take screenshots or access game memory somehow OR use
a recreated game -- modifying the source code.

3. how to take screenshots & clip it to the emulator in a fixed position

4. how to record many screenshots over the course of a playthrough(should be grayscale, and small size)

5. How to record controller via computer

6. How to record controller input at time of screenshot being taken

7. training data will take the form {screenshot, controller input} & should be stored locally somehow

8. Use this data to train neural network

9. Figure out a way to load preset weights / nodes as a network to avoid learning every time -- can save the "best" network found.

~~~~~~~~~~~PHASE II~~~~~~~~~~~

To run the AI, feed screenshot of it playing to NN, get output to determine the decision. Based on the decision, a control is made.

Fortunately feeding forward is a fast computation. How fast is it to preprocess the data before input at real time? This may be a source of slowdown. Truly, TensorKart only really wanted the GPU for the learning process -- which admittedly may take a long time if you have possibly thousands of screenshots and many layers to train.
