# Model Zoo

I provide finetuned DETR models, and plan to include more in future. AP is computed on PASVOC 2012val

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
      <td><b>67.5<b></td>
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
