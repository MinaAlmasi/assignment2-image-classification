After running ```preprocess_data.py```, this folder will contain a compressed, preprocessed version of the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). This file was purposely not uploaded to git (but rather  ```*.npz``` files are in  ```.gitignore```) due to its large size. 

 See also, the [*Pipeline*](https://github.com/MinaAlmasi/assignment2-image-classification/tree/master#pipeline) section for instructions on how to run the entire classification pipeline. **If you are only interested in preprocessing the data, you can run:** 

```
python src/preprocess_data.py
```

The preprocessing involves greyscale conversion, scaling and reshaping the image data. The data is saved as numpy arrays: 

* X_train_feats, X_test_feats (preprocessed image data)
* y_train (labels 0-9)
* y_test  (labels 0-9)
* labels  (explicit labels: "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
