**DETR**: End-to-End Object Detection with Transformers
========
PyTorch training code and pretrained models for **DETR** (**DE**tection **TR**ansformer).
DETR replace the full complex hand-crafted object detection pipeline with a Transformer, and match Faster R-CNN with a ResNet-50, obtaining **68 AP** on PASCAL VOC 2012. Inference in 50 lines of PyTorch.

![DETR](.github/DETR.png)

**What it is**. Unlike traditional computer vision techniques, DETR approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. 
Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.

**About the code**. I believe that object detection should not be more difficult than classification,
and should not require complex libraries for training and inference.
DETR is very simple to implement and experiment with, and I provide a
[Colab Notebook](https://colab.research.google.com/drive/1OVwwdWi6C7kje_k_8vtZVL7_3s5a5s7v?authuser=4#scrollTo=GOmj90akyjKB)
showing how to training & inference with DETR.

For details see [End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers)

# Model Zoo

I provide finetuned DETR models, and plan to include more in future.
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
      <td>DETR-R50 Finetuned</td>
      <td>DETR-R50R</td>
      <td>40</td>
      <td>4:03:58</td>
      <td>A100</td>
      <td>67.5</td>
      <td>50.1</td>
      <td>47.3</td>
      <td>10.7</td>
      <td>30.2</td>
      <td>59.5</td>
      <td><a href="https://drive.google.com/file/d/1-Pm-eDlvVr0Is4dx1LkcGobhsFIPb4RT/view?usp=share_link">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1-YHYYsRtEyR9uitMui24p0qEyfZxgTKS/view?usp=share_link">logs</a></td>
      <td>474,1 MB</td>
    </tr>
  </tbody>
</table>

```python
# Load fine-tuned model via checkpoint 
num_classes = 22  # PV12: 20 + noObject (1)

model = torch.hub.load('facebookresearch/detr',
                       'detr_resnet50',
                       pretrained=False,
                       num_classes=num_classes)

checkpoint = torch.load('outputs/checkpoint.pth',
                        map_location='cpu')

model.load_state_dict(checkpoint['model'],
                      strict=False)
```

## Tracking Loss 
Solid lines for training and dashed line for validation results

| ![Architecture](https://github.com/tuanlda78202/DLP/blob/master/df/materials/lossmAP.png) | 
|:--:| 
| Total loss & mAP 40e|

| ![Architecture](https://github.com/tuanlda78202/DLP/blob/master/df/materials/cebboxgiou.png) | 
|:--:| 
| Loss CE, BB & GIoU 40e|

| ![Architecture](https://github.com/tuanlda78202/DLP/blob/master/df/materials/clcaerror.png) | 
|:--:| 
| Class Error & Cardinality Error Unscaled 40e|
