from data_loading import *
import torch
import numpy as np
from unet import UNet
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
frame = None
frame = AppendtoFrame()

def compute_iou():
    # IoU test of the model

    weights = 'best_caries.pth'
    net = UNet(n_channels=1, n_classes=1).cuda()

    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
    else:
        print('weights not loaded')
        return

    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((400, 500)), transforms.ToTensor()])

    #change interval for desired images
    test = frame.loc[0:82]
    img_path_list = test['img_path']
    mask_path_list = test['mask_path']

    avg_iou = 0

    image = cv2.imread(img_path_list[0], cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[0], image.shape[1]

    flag_good = 0
    flag_normal = 0
    flag_low = 0
    flag_bad = 0
    net.eval()
    with torch.no_grad():
        for i in range(len(img_path_list)):
            image = cv2.imread(img_path_list[i], cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path_list[i], cv2.IMREAD_GRAYSCALE)
            input = transform(image).cuda()
            input = torch.unsqueeze(input, dim = 0)
            output = net(input)
            output = output.cpu().numpy().squeeze(0).transpose((1, 2, 0))
            output = cv2.resize(output,(w, h))
            output = np.clip(output, 0, 1)
            etVal, output = cv2.threshold(output, 0, 1, cv2.THRESH_BINARY)
            iou = np.sum(np.logical_and(output, mask)) / np.sum(np.logical_or(output, mask))
            
            if iou >= 0.8:
                flag_good+=1
            elif iou >= 0.5:
                flag_normal+=1
            elif iou >= 0.2:
                flag_low+=1
            else:
                flag_bad+=1
            print("Sample ", i+1, ": ", iou)
            avg_iou += iou
    
    print(i+1, " sample tested \n")
    print("Final avg_iou: ", avg_iou/(i+1))
    
    print("\nbad :", flag_bad)
    print("\nlow :", flag_low)
    print("\nnormal :", flag_normal)
    print("\ngood :", flag_good)

    y = np.array([flag_bad, flag_low, flag_normal, flag_good])
    x = np.array(["0~0.2","0.2~0.5", "0.5~0.8", "0.8~1.0"])
    plt.pie(y,
            labels = x,
            colors=["#d5695d", "#5d8ca8", "#65a479", "#a564c9"],
            explode=(0.2, 0.2, 0, 0), 
            autopct='%.2f%%',
    )
    #plt.bar(x, y, alpha=0.9, width = 0.35, facecolor = 'lightskyblue', edgecolor = 'white', lw=1)
    plt.title("IoU Pie Chart")
    plt.savefig(".\\train\\IoU_pie.jpg")
    plt.show()

def get_stackbarchart():
    # Draw stack bar

    X = ['All data','Generated','Drawn','Natural']
    bad = np.array([6,5,1,0])
    low = np.array([13,9,2,2])
    normal = np.array([30,22,7,1])
    good = np.array([37,36,1,0])

    plt.bar(X, bad, color='r')
    plt.bar(X, low, bottom = bad, color='y')
    plt.bar(X, normal, bottom = bad + low, color='b')
    plt.bar(X, good, bottom = bad + low + normal, color='g')
    plt.xlabel("Data split")
    plt.ylabel("IoU value")
    plt.legend(["0<IoU<0.2", "0.2<IoU<0.5", "0.5<IoU<0.8", "0.8<IoU<1.0"])
    plt.title("IoU of different data")
    plt.show()

if __name__ == '__main__':
    #compute_iou()
    #get_stackbarchart()
    pass