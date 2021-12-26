#!/opt/homebrew/Caskroom/miniforge/base/envs/pytorch/bin/python
# Script to classify a singular MRI as having a tumor or not having one
import getopt
from os import error
import sys
import time
from PIL import Image
import torch
from torchvision import transforms as T
import torch.cuda as cuda

def main(argv):
    try:
        opts, args = getopt.getopt(argv,'hm:i:')
    except getopt.GetoptError:
        print('Usage:classify_image.py -m <model> -i <image>')
        sys.exit(2)
    
    # Parse command-line arguments
    for opt,arg in opts:
        if opt == '-h':
            print('Usage: classify_image.py -m <model> -i <image>')
            sys.exit()
        elif opt in ('-m', '--model'):
            # Check for CUDA
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                print('Running on ' + cuda.get_device_name(device) + '...') 
                model = torch.load(arg, map_location='cuda:0')
                model.eval()
            else:
                device = torch.device('cpu')
                print('Running on the CPU...')
                model = torch.load(arg, map_location=torch.device('cpu'))
                model.eval()
        elif opt in ('-i', '--image'):
            image = Image.open(arg)

    if(len(argv) != 4):
        print('Usage: classify_image.py -m <model> -i <image>')
        sys.exit(2)

    # Image processing
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()])
    image = image.convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)

    # Check for CUDA
    if torch.cuda.is_available():
        image = image.cuda()
    else:
        image = image.clone().detach().requires_grad_(False)
    
    # Set tensor to run on device
    image = image.to(device)

    # Feed image through network and get prediction
    with torch.no_grad():
        pred = model(image)

    # Output the prediction
    if(pred.numpy().argmax() == 0) : output = 'tumor not detected'
    elif(pred.numpy().argmax() == 1) : output = 'tumor detected'
    print('Predicted output: ' + output)

if __name__ == '__main__':
    start_time = time.time()
    main(sys.argv[1:])
    print('Runtime:', time.time() - start_time, 'seconds')