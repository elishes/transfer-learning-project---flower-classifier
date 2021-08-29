# Transfer learning project - flower sorter
the task at hand was to build, through transfer learning, models which could accuractely classify flowers into their correct categories (dandelions, roses, etc). 
Three visual recognition models were used for this task - VGG-11, ResNet-18, and DenseNet121, all in PyTorch. 

# Preprocessing:
In the preprocessing stage, the images were imported in batches of fifty and enriched with randomized resizing, cropping, and flipping; the enrichment was conducted to increase the difficulty of the learning task, with the goal of preventing overfitting. Two train-validation-test splits were performed: 40-30-30 (respectively) and 50-25-25 (respectively). All models were run on both types of data splits. Results will be presented per model and per data split.

# The models:
The modifications made to the pretrained models were similar across all models. The final fully connected layer in the pretrained models were replaced with a new fully connected layer, containing the new task’s number of training classes and the trained weights of the new task. All models were trained with fifty epochs.

I found that DenseNet121 performed better than VG-11 or ResNet-18, with 0.93 accuracy for the 50-25-25 split.

See PDF for full report, graphs etc.
