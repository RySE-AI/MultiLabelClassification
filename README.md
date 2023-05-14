# Mutli-Label Image Classification with Pytorch Lightning

The purpose of this project is to learn and implement a multi-label image
classification model by using Lightning. The focus of this project will not be 
solely on the classifier itself but rather on the implementation process and
associated considerations.
Additionally, the project will involve integrating MLFlow and potentially 
Neptune to track and monitor model training progress and metrics.
To visualize my inference of the model I will use a jupyter notebook and will 
create a streamlit app aswell.

### Why Multilabel Classfication?
First and foremost I want to learn how to use several modules like lightning, 
mlflow/neptune. It was kind a random to choose this task but if offers a 
learning opportunity for myself.
Of course you can use an Object Detection Model to detect several objects in an
image but sometimes it's faster for me to use torch's ImageFolder style labeling
instead of using CVAT (not considering the SAM implementation). So I adapted the
torchvision ImageFolder class for Multi-Label Classification tasks.

# <u>Topic/Task</u>

- Create an image classifier with PyTorch Lightning capable of classifying more than one class per image.
  - Sort and label the data using the ImageFolder style
  - Utilize transfer learning with a classic ResNet50 backbone or similar architecture
- Track the metrics of training and validation with MLFlow and Neptune
- Visualize the model's inference on a test dataset
  - Display metrics such as a confusion matrix
  - Provide visual predictions in both the notebook and Streamlit app
- Perform a benchmark comparison to an object detection model
  - Evaluate metrics
  - Measure labeling time
  - Analyze training time
  - Assess model size and inference time

### <u>Hardware Specs: </u>

MacBook Pro (16'', 2021) <br />
Apple M1 Max <br />
32 GB Memory <br />
Ventura 13.3.1 (a)

### <u>Tools:</u>

Environment Manager: Conda (I will use Poetry aswell - learning purpose) <br />
Editor: PyCharm CE <br />
Terminal: zsh <br />

### <u>Main Tech:</u>
Pytorch <br />
Lightning <br />
MLFlow <br />
Streamlit <br />
Jupyter Notebook <br />
CVAT

