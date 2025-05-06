# Vehicle Detection with Yolov5

<details open>
<summary><strong style="font-size: 1.5em;">1. Project overview</strong></summary>

- The aim of this project is to apply the [YOLOv5](https://github.com/ultralytics/yolov5) model for object detection
by continuing the training of the model on a specific vehicle dataset. [YOLOv5](https://github.com/ultralytics/yolov5) 
is one of the most popular and high-performing models for computer vision tasks such as object classification, 
detection, and segmentation.
- About the dataset: see [2. Dataset](#2)
- About the pretrained model: see [3. Model](#3)
- This project is only for private educational purpose.

</details>

<details open>
<a id="2"></a>
<summary><strong style="font-size: 1.5em;">2. Dataset</strong></summary>

- The vehicle dataset used for training in this project includes 8 classes: auto, bus, car, lcv, motorcycle, multiaxle, tractor, truck. It was provided by Saksham Jain
and is available on Kaggle at the following link: [Vehicle Detection 8 Classes - Object Detection](https://www.kaggle.com/datasets/sakshamjn/vehicle-detection-8-classes-object-detection/data)
- The original dataset was downloaded and saved under the following directory: `dataset/verhicles_8/train`
- The training, validation and test datasets are split from the original dataset and save under the following directory
`dataset/verhicles_8/training_yolov5`. The number of images in each dataset is 6000, 2000 and 218, respectively.
The required information of dataset is saved in the file `vehicles8.yaml`, which is used later for training.  
- The script file `dataset.py` is responsible for data processing, which helps to split the original dataset into 
separate datasets for training and testing.

</details>


<details open>
<a id="3"></a>
<summary><strong style="font-size: 1.5em;">3. Model</strong></summary>

- The YOLOv5 repository is available on Ultralytics' GitHub: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5). 
The YOLOv5 repository is cloned locally for training but is not be pushed on this repository.
- The model used for training in this project is the pretrained model YOLOv5s.
- In order to clone YOLOv5 repository for continuing training, follow the below commands on terminal. 

```bash
# Source: https://github.com/ultralytics/yolov5
# Clone the YOLOv5 repository
git clone https://github.com/ultralytics/yolov5

# Navigate to the cloned directory
cd yolov5

# Install required packages
pip install -r requirements.txt
```

- To clone this repository (vehicle-detection-yolov5), follow the below command and then the commands above

```bash
# Clone the repository
git clone https://github.com/hai-van-tran/vehicle-detection-yolov5
```


</details>

<details open>
<summary><strong style="font-size: 1.5em;">4. Training</strong></summary>

- To train the detection model using the pretrained weights (in this case `yolov5s.pt`), follow the below commands.
The batch size and the number of epochs can be changed according to the technical availability and preference. 
See [5. Result](#5) for the training results with the different number of epochs.

```bash
# Source: https://github.com/ultralytics/yolov5

# navigate to the `yolov5` directory
cd yolov5

# start training with pretrained weights yolov5s.pt for 10 epochs
python train.py --img 640 --batch 16 --epochs 10 --data ./vehicles.yaml --weights yolov5s.pt
```

- To continue training the model using the best weights from a previous training run,
replace the `--weights yolov5s.pt` option with the path to the best weights, for example `--weights ./save_model/best.pt`

</details>

<details open>
<summary><strong style="font-size: 1.5em;">5. Result</strong></summary>

- Within this project, there were 2 trainings with different epoch numbers executed. The results are normally 
automatically saved in the directory `yolov5/runs/train/exp` (the directory `yolov5` exists after the YOLOv5 repository is cloned.
- In this project, two training runs were executed, with the same pretrained weights, batch size and dataset. 
The only difference was the number of epochs: one with 10 epochs, while the other 100.
- The results of training the model for 10 epochs are in the directory `result/train/exp`, 
while the results of training for 100 epochs are in `result/train/exp2`.

</details>

<details open>
<summary><strong style="font-size: 1.5em;">6. Inference</strong></summary>

- In order to apply the model on detecting object, follow the below commands. 
Output images after object detection are normally automatically saved in the directory `yolov5/runs/detect`.
To use different weights for the model, replace the `--weights ../save_model/best.pt` option with the path to 
the desired weights.

```bash
# Source: https://github.com/ultralytics/yolov5

# navigate to the `yolov5` directory
cd yolov5

# start training with pretrained weights yolov5s.pt for 10 epochs
python detect.py --weights ../save_model/best.pt --source ../dataset/vehicles_8/training_yolov5/images/test
``` 
- The inference output using model trained for 10 epochs is in the directory `result/detect/exp`, and 
the output using model trained for 100 epochs is in `result/detect/exp2`. 
In general, training the model for 100 epochs results in better performance, including more accurate object detection 
and bounding box localization.
- However, the model trained for 100 epochs is overfitting, soo it falls to detect any vehicles in the different scenarios.
In contrast, the model trained for 10 epochs can still detect some cases correctly.

</details>
