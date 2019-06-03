## Creating a Sentiment Analysis Web App Using PyTorch and SageMaker

Amazon SageMaker provides every developer and data scientist with the ability to build, train, and deploy machine learning models quickly. Amazon SageMaker is a fully-managed service that covers the entire machine learning workflow to label and prepare your data, choose an algorithm, train the model, tune and optimize it for deployment, make predictions, and take action. Your models get to production faster with much less effort and lower cost.

Our goal, in this project, is to have a simple web page which a user can use to enter a movie review. The web page will then send the review off to our deployed model which will predict the sentiment of the entered review.

### General Outline
General outline for SageMaker projects using a notebook instance includes the following steps.

- Download or otherwise retrieve the data.
- Process / Prepare the data.
- Upload the processed data to S3.
- Train a chosen model.
- Test the trained model (typically using a batch transform job).
- Deploy the trained model.
- Use the deployed model.

For this project, we will be following the steps in the general outline with some modifications.

First, we will not be testing the model in its own step. We will still be testing the model, however, we will do it by deploying our model and then using the deployed model by sending the test data to it. One of the reasons for doing this is so that we can make sure that our deployed model is working correctly before moving forward.
In addition, we will deploy and use our trained model a second time. Our newly deployed model will be used in the sentiment analysis web app.

### Dataset and Preprocessing
We will be using the [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/). To begin with, we will read in each of the reviews and combine them into a single input structure. Then, we will split the dataset into a training set and a testing set. 
Now that we've read the raw training and testing data from the downloaded dataset, we will combine the positive and negative reviews and shuffle the resulting records.

The first step in processing the reviews is to make sure that any html tags that appear should be removed. In addition we wish to tokenize our input, that way words such as entertained and entertaining are considered the same with regard to sentiment analysis.

### Transforming the data
To start, we will represent each word as an integer. Of course, some of the words that appear in the reviews occur very infrequently and so likely don't contain much information for the purposes of sentiment analysis. The way we will deal with this problem is that we will fix the size of our working vocabulary and we will only include the words that appear most frequently. We will then combine all of the infrequent words into a single category and, in our case, we will label it as 1.

Since we will be using a recurrent neural network, it will be convenient if the length of each review is the same. To do this, we will fix a size for our reviews and then pad short reviews with the category 'no word' (which we will label 0) and truncate long reviews.


### Upload the data to S3
We will need to upload the training dataset to S3 in order for our training code to access it. 
It is important to note the format of the data that we are saving as we will need to know it when we write the training code. In our case, each row of the dataset has the form label, length, review[500] where review[500] is a sequence of 500 integers representing the words in the review.

### Build and Train the PyTorch Model
In the SageMaker framework, a model comprises three objects
- Model Artifacts
- Training Code
- Inference Code
each of which interact with one another. Here we will still be using containers provided by Amazon with the added benefit of being able to include our own custom code.

We will start by implementing our own neural network in PyTorch along with a training script ```model.py```. When a PyTorch model is constructed in SageMaker, an entry point must be specified. This is the Python file which will be executed when the model is trained. Inside of the train directory is a file called ```train.py``` which contains most of the necessary code to train our model.

### Deploy the model for testing
After we have trained our model, we would like to test it to see how it performs. Our model will take input of the form review_length, review[500] where review[500] is a sequence of 500 integers which describe the words present in the review, encoded using word_dict. Fortunately for us, SageMaker provides built-in inference code for models with simple inputs such as this.

### Use the model for testing
Once deployed, we can read in the test data and send it off to our deployed model to get some results. Once we collect all of the results we can determine how accurate our model is.

### Deploy the model for the web app
After we know that our model is working, it's time to create some custom inference code so that we can send the model a review which has not been processed and have it determine the sentiment of the review.

By default the estimator which we will create, when deployed, will use the entry script and directory which we provided when creating the model. However, since we would like to accept a string as input and our model expects a processed review, we need to write some custom inference code.

We will store the code that we write in the serve directory. Provided in this directory is the ```model.py``` file that we will use to construct our model, a ```utils.py``` file which contains the review_to_words and convert_and_pad pre-processing functions which we used during the initial data processing, and ```predict.py```, the file which will contain our custom inference code. Note also that requirements.txt is present which will tell SageMaker what Python libraries are required by our custom inference code.

When deploying a PyTorch model in SageMaker, we are expected to provide four functions which the SageMaker inference container will use.

- model_fn: This function is the same function that we used in the training script and it tells SageMaker how to load our model.
- input_fn: This function receives the raw serialized input that has been sent to the model's endpoint and its job is to de-serialize and make the input available for the inference code.
- output_fn: This function takes the output of the inference code and its job is to serialize this output and return it to the caller of the model's endpoint.
- predict_fn: The heart of the inference script, this is where the actual prediction is done and is the function which you will need to complete.

For the simple website that we are constructing during this project, the input_fn and output_fn methods are relatively straightforward. We only require being able to accept a string as input and we expect to return a single value as output. We might imagine though that in a more complex application the input or output may be image data or some other binary data which would require some effort to serialize. At the end we will test the model and will use the model in webapp. 

For implementation details, please refer to [this file]( https://github.com/UsmanIjaz/DL_Sagemaker/blob/master/SageMaker%20Project.ipynb)



