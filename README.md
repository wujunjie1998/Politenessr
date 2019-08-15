# Politenessr

## Intro
Politenessr is a package used to predict the value of politeness of texts.

It is based on a fine tuned BERT model.
## Install 

### Use pip
If `pip` is installed, politenessr could be installed directly from it:

    pip install politenessr

### Dependencies
	python>=3.6.0
	torch>=0.4.1
	numpy
	pandas
	unidecode
	pytorch-pretrained-bert
	pytorch-transformers
	


## Usage and Example

### Notes: During your first usage, the package will download a model file automatically, which is about 400MB.

### `predict`
`predict` is the core method of this package, 
which takes a single text of a list of texts, and returns a list of raw values in `[1,5]` (higher means more politeness, while lower means less).

### Simplest usage

You may directly import `politenessr` and use the default predict method, e.g.:

    >>> import politenessr
    >>> politenessr.predict(["I am totally agree with you"])
    [4.3568916]
    
### Construct from class
Alternatively, you may also construct the object from class, where you could customize the model path and device:
 
	>>> from politenessr import Politenessr
	>>> pr = Politenessr()
	
	# Predict a single text
	>>> pr.predict(["I am totally agree with you"])
	[3.5638056]
	
	# Predict a list of texts
	>>> preds = pr.predict(['I am totally agree with you','I hate you'])
    >>> f"Raw values are {preds}"
    [3.5638053 2.2007465]



More detail on how to construct the object is available in docstrings.

### Model using multiprocessing when preprocessing a large dataset into BERT input features 
If you want to use several cpu cores via multiprocessing while preprocessing a large dataset, you may construct the object via

    >>> pr = Politenessr(CPU_COUNT=cpu_cpunt, CHUNKSIZE=chunksize)

If you want to faster the code through multi gpus, you may construct the object via

    >>> pr = Politenessr(is_paralleled=True, BATCH_SIZE = batch_size)


## Contact
Junjie Wu (wujj38@mail2.sysu.edu.cn)
