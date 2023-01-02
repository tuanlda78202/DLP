# A SSD model with FPN

As an algorithm with better detection accuracy and speed, SSD (Single Shot MultiBox Detector) has made great progress in many aspects. However, it can’t achieve a good detection efect for small objects because
it does not make full use of high-level semantic information. For that reason, we propose a SSD model with FPN
to solve this problem, which can reduce parameters, narrow internal space and improve performance for
small objects. As revealed from the experimental results, our model was trained for 280 epochs, obtaining the mAP of 63.8% on the VOC PASCAL 2012 dataset.

## Usage
To load the model
```python
checkpoint = "checkpoint\checkpoint_ssd300_280.pth.tar"
# Load model checkpoint 
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
```
To detect an image
```python
from PIL import Image
from detect import detect
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'checkpoint\checkpoint_ssd300_280.pth.tar'
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
  
image = Image.open(image_file).convert('RGB') 
detect(image, min_score=0.25, max_overlap=0.5, top_k=200)
```
## Results
AP is computed on PASVOC 2012val

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>schedule</th>
      <th>training time</th>
      <th>gpu training</th>
      <th>AP@0.5</th>
      <th>AP@0.75</th>
      <th>AP@0.5:0.95</th>
      <th>AP@small</th>
      <th>AP@medium</th>
      <th>AP@large</th>
      <th>url</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FPN-SSD</td>
      <td>Fpn-resnet50</td>
      <td>280</td>
      <td></td>
      <td>P100</td>
      <td>63.8</td>
      <td>35.0</td>
      <td>35.6</td>
      <td><b>14.3<b></td>
      <td></td>
      <td></td>
      <td><a href="https://drive.google.com/file/d/1IiJEYI58f4nTyFmca-mdZ2V1hpF5T7lG/view?usp=sharing">model</a>&nbsp</td>
      <td>267 MB</td>
    </tr>
  </tbody>
</table>

## Examples
Below are some detection results:

![Dec1](https://github.com/tuanlda78202/DLP/blob/master/ssd/FPN-SSD/detected_img/detect1.png "Detection 1")
![Dec2](https://github.com/tuanlda78202/DLP/blob/master/ssd/FPN-SSD/detected_img/detect2.png "Detection 2")
![Dec3](https://github.com/tuanlda78202/DLP/blob/master/ssd/FPN-SSD/detected_img/detect3.png "Detection 3")
![Dec4](https://github.com/tuanlda78202/DLP/blob/master/ssd/FPN-SSD/detected_img/detect4.png "Detection 4")
