import time
from models.fpnssd300 import SSD300, MultiBoxLoss
import torch 
from utils.utils import adjust_learning_rate, save_checkpoint, AverageMeter, clip_gradient

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

label_map = {v: k for k, v in rev_label_map.items() if k!= 0}

grad_clip = 1
print_freq = 50

def train(checkpoint, train_data_loader, lr, momentum, weight_decay, iterations, train_dataset, decay_lr_to, decay_lr_at):
    """
    Training.
    """
    global label_map, epoch

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes = 21)
        for index, child in enumerate(model.base.resnet.children()):
            if index != 7:
                for param in child.parameters():
                    param.requires_grad = False
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum = momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    epochs = iterations // (len(train_dataset) // 16)
    print("Number of epochs: ", epochs)
    decay_lr_at = [it // (len(train_dataset) // 16) for it in decay_lr_at]
    
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train_one_epoch(train_loader=train_data_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        print("Saving checkpoint epoch:", epoch)
        save_checkpoint(epoch, model, optimizer)


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, test_data_loader):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    losses = AverageMeter()  # loss


    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        
        start = time.time()
        
        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Training Time {3:.3f} \t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  (time.time()-start)*print_freq, loss=losses))
    
    if epoch % 10 == 0:
        model.eval()
        val_losses = AverageMeter()

        with torch.no_grad():
            # Batches
            for i, (images, boxes, labels, difficulties) in enumerate(test_data_loader):
                images = images.to(device)  # (batch_size (N), 3, 300, 300)
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]

                # Forward prop.
                predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

                # Loss
                loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

                val_losses.update(loss.item(), images.size(0))

            # Print status
            print('Validation loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=val_losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored