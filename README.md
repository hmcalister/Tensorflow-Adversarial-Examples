# Tensorflow Adversarial Example Training

This project is a spin off of a [much larger research project](https://github.com/hmcalister/Tensorflow-Intepretability-Project) undertaken at the University of Otago. This project was small part of the larger whole, but is interesting and independent enough to be preserved alone.

The goal of this project is to train a neural network using adversarial examples, generated using gradient *ascent* for each image in a training dataset. This is done to try and improve an interpretability method labelled `maximal_class_activations`. In this interpretability method we use backpropagation to iterate random noise towards an image that maximally activates a specific output of the network. This, in theory, should produce an image that is a perfect example of whatever class we chose. In practice, neural networks are not perfect and tend to classify random noise as "perfect" examples of classes. Adversarial examples are created in a very similar way to this interpretability method, just starting with a training image not random noise and aiming to minimize class activation rather than maximize. We hope that by training on adversarial examples we can eliminate the noisy images produced by `maximal_class_activations` and instead approach a better representation of the network behavior.

## Files

`AdversarialExamples.ipynb` is a notebook with the interpretability methods in use.

`AdversarialTraining.py` is a script to train a convolutional neural network on both vanilla images and adversarial examples in an interleaved manner. Training begins with with a set of vanilla epochs, followed by a set of adversarially generated epochs. This cycle is repeated for as long as specified (or can be cut short using `CTRL+C`). The model from this training is saved to disk, as is the training history from all epochs (vanilla and adversarial). Warning: creating adversarial examples can be very slow!

`AdversarialTrainingHistoryProcessing.py` is a short script to take the saved dataframe from `AdversarialTraining.py` and plot it nicely, including options to indicate vanilla and adversarial epochs, although this is usually obvious.

## Graphics

### Loss During Training

![Graph of loss and validation loss over an adversarial training period](images/loss2.png)

The large spikes every 10 epochs is the interruption due to an adversarial epoch. It is also unfortunate that validation loss increased so wildly - I think the additional adversarial examples may have allowed the network to overfit the training data even more than is usually possible?

One solution, especially for simple tasks like MNIST, is to not train for so long! About 20 epochs is good enough!

### Adversarial Examples

![Some images produced using the adversarial example technique detailed above](images/adversarialExample.png)

For MNIST, which is greyscale, an adversarial example is simply a spattering of monocolored noise across the image! In your trials, you may notice there are several images with seemingly no noise. This occurs when a model has massively overfit the training data - so much so that tensorflow gave up on assigning a gradient. There is effectively no way to change these images to trick the model (within the bounds of a 64 bit integer used by tensorflow graphs).

### Maximal Class Activations

![The classes of MNIST (digits 0-9) as predicted by a gradient ascent through the model](images/maximalClassActivation.png)

Finally, we have some examples of maximal class activations. These are still not great (else I would already be a millionaire) but on closer inspection there appears to be some structure we expect. The image for 2 seems to have a slight ghost of a 3, as does 4, 6, 8, and 9. The other classes may be reduced to random noise, or perhaps are taking gradients from so many filters that they are effectively combinations of every *possible* digit in that class, rather than a prototypical digit.

We can improve these images by adding some constraints on the generated tensors. If we add blur every few steps we implicitly penalize (explicitly remove) high frequency terms - which removes the fuzziness over the image we see above. If we also every so often move the image around by a few pixels we can avoid the images being true adversarial examples - better picking up the features intended. An example of the images produced with both of these effects is below:

![Maximal Class Images with added constraints during creation](images/betterMaximalClassActivation.png)
