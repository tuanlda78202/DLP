from torchvision import transforms
import torch
from PIL import Image, ImageDraw, ImageFont
from utils.utils import vis_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rev_label_map= {0: 'background',
                1:'aeroplane',
              2:'bicycle',
              3:'bird',
              4:'boat',
              5:'bottle',
              6:'bus',
              7:'car',
              8:'cat',
              9:'chair',
              10:'cow',
              11:'diningtable',
              12:'dog',
              13:'horse',
              14:'motorbike',
              15:'person',
              16:'pottedplant',
              17:'sheep',
              18:'sofa',
              19:'train',
              20:'tvmonitor'}
              
# Load model checkpoint
checkpoint = '..\checkpoint\checkpoint_ssd300_280.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    with torch.no_grad():
        # Transform
        image = normalize(to_tensor(resize(original_image)))

        # Move to default device
        image = image.to(device)

        # Forward prop.
        predicted_locs, predicted_scores = model(image.unsqueeze(0))

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                                 max_overlap=max_overlap, top_k=top_k)

        # Move detections to the CPU
        det_boxes = det_boxes[0].to('cpu')

        # Transform to original image dimensions
        original_dims = torch.FloatTensor(
            [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
        det_boxes = det_boxes * original_dims

        # Decode class integer labels
        det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

        # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
        if det_labels == ['background']:
            # Just return original image
            return original_image

        # Suppress specific classes, if needed
        for i in range(det_boxes.size(0)):
            if suppress is not None:
                if det_labels[i] in suppress:
                    continue
    vis_image(original_image, det_boxes, det_labels, scores=det_scores[0])

