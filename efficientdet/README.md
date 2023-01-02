# References
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch : use this tool for implement EfficientDet-DO.
https://github.com/yukkyo/voc2coco: convert PASCAL VOC format into COCO format.

# Model Zoo

I provide finetuned EfficientDet-D0 models, and plan to include more in future. AP is computed on PASVOC 2012val

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>schedule</th>
      <th>AP@0.5</th>
      <th>AP@0.75</th>
      <th>AP@0.5:0.95</th>
      <th>AP@small</th>
      <th>AP@medium</th>
      <th>AP@large</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EfficientDet-D0 Finetuned</td>
      <td>EfficientNet-B0</td>
      <td>30</td>
      <td><b>66.4<b></td>
      <td>50.7</td>
      <td>46.5</td>
      <td>11.5</td>
      <td>32.8</td>
      <td>55.8</td>
    </tr>
  </tbody>
</table>
