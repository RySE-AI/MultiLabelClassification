# Mutli-Label Image Classification with Pytorch Lightning

The purpose of this project is to learn how to implement a multi-label image
classification model by using Lightning. The focus of this project will not be 
solely on the classifier itself but rather on the implementation process and
associated considerations.
Additionally, the project will involve integrating MLFlow to track and monitor 
the model training progress and metrics.
To visualize my inference of the model I will use jupyter notebooks and will 
create a streamlit app as well.

### <u>Why multi-label classfication?</u>
First and foremost I want to learn how to use several modules like lightning, 
MLFlow and streamlit. It was kind a random to choose this task but if offers a 
learning opportunity for myself.
I believe that for a manageable amount of classes it can be faster to
use torch's ImageFolder style labeling instead of using labeling tools
like CVAT. So I adapted the torchvision ImageFolder class for multi-label
classification tasks. For an unknown amount of classes — or a high amount for
that matter — i would recommend using CVAT or similar tools.

### <u>What's the difference to multi-class classification?</u>
- Multi-class: Predicting one class per image/data
- Multi-label: Predicting one or more class/label per image/data

**_Visual concept of multilabel:_** <br/>
<img src="img/NeuralNetwork_MultiLabel_Concept.svg" width=500 height=400 align="center"/>

### <u>Project Summary</u>
**Important: Content, both literal and graphical, is copyrighted by Wizards of
the Coast. This is a fan project with non-comercial usage!**

Recently I was at my parents home and I randomly saw my old magic the gathering
cards. I thought that this would be good and comprehensible example for a multi-label
classifier. So I made my own example by using Magic the Gathering card images from
mtgpics. I guess that there is no real life use case for it but as I mentioned
before this project is for learning purposes. 

SO the task will be to predict the color of the card and whether it's a creature
or a special (spell/instant/enchantment) with one classifier.

To get the images I built a [webscraper](scripts/mtgpics_card_scraper.py) with
request and bs4. I scraped the images of several card-decks and sorted them into
different folders, like this:

<img src="img/folder_structure.png" width=500/>

It took about **~6.5 mins** to sort the images of a whole deck (~250-300 images) 
and to make a quick check afterwards. Take a look at the [data exploration notebook](notebooks/00_Image_Exploration.ipynb)
for more details about the data/images.

Followed by the data exploration I trained 3 different models in 
[this notebook](notebooks/01_Training_with_Lightning.ipynb) and tracked their 
results on a local MLFlow instance. In all experiments I used a Resnet18 as my
backbone and I trained the models on my M1 GPU.

<img src="img/MLFlow_tracking_results_table.png" width=700/>

1. Experiment: pretrained backbone + frozen weights
2. Experiment: pretrained backbone
3. Experiment: untrained backbone + normalized on own train dataset

The 2nd experiment leads to the best results in Accuracy, Precision, Recall. 
So I used the best checkpoint and performed some
[visual inference](notebooks/02_Model_Inference.ipynb) on the test dataset.
Furthermore I used streamlit to create a little prediction app, where you're 
able to pass any image to it and get a confidence score to each class (sigmoid prediction).

<img src="img/streamlit_example.png" width=700/>

The model is able to make prediction on real cards as well but as expected it's
not so good as on mtgpic's images. This is one of the good examples ;)


# Run the streamlit app with docker

I pushed the image to [Dockerhub](https://hub.docker.com/repository/docker/ryseai/magic_the_gathering_streamlit/general).
To run the container, use the following command in your terminal:

```bash
docker run -p 8501:8501 ryseai/magic_the_gathering_streamlit:v0.1-dev
```
It should display this:

<img src="img/streamlit_test/docker_run_streamlit_app.png" width=700/>

`-p 8501:8501`: is the default port of streamlit

If you start the container, you can access the app via http://0.0.0.0:8501.

Attention: If you doesn't mount your torch .cache it will download the resnet18
first. So that's why the load_model() step at beginning will take some time. 
Look at the streamlit docs for further [explanations](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
about the deployment with docker.

*Attention 2:* I build the container for arm64 archs. It works on a amd64 but
it's really slow! I'll try to look up how to setup a multi plattform container!

# Information

Feel free to ask me if you have any questions to my projects or if you have 
suggestions for improvement. :)

### TODOs
- Upload the sorted data somewhere(Suggestions? Is it allowed to upload the data (copyrights?))
  - maybe create a shell script to download the data I used and sort them...
  - or upload the data to lfs?
- Finish documentation/docstrings
- Use captum for explainable AI on some examples
- test streamlit app on another laptop/pc/os



# Setup Details
## <u>Hardware Specs</u>

MacBook Pro (16'', 2021) <br />
Apple M1 Max <br />
32 GB Memory <br />
Ventura 13.3.1 (a)

## <u>Tools</u>

Python Manager: Conda/pip <br />
Editor: PyCharm CE and VSCode <br />
Notes: Obsidian <br />
Terminal: zsh <br />
Package Manager: Homebrew

I will use poetry in the future but I had less problems with setting my
environment with conda on M1(MPS).
## <u>Main Tech:</u>
see requirements.txt <br />
Pytorch <br />
Lightning <br />
MLFlow <br />
Streamlit <br />
Jupyter Notebook <br />
