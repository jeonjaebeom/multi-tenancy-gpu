#!/usr/bin/python3

from concurrent.futures import process
from urllib.parse import uses_query
import torchvision.models as models
import torch
import cv2
import argparse
import time, os
import sys
# from random import random
from math import factorial, exp
# from utils import (load_alexnet_model, preprocess, read_classes)
from multiprocessing import Process, Queue, Value
from torchvision import transforms
import numpy as np
import statistics as st
from numpy import random

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='input/bus.jpg', 
                    help='path to the input image')
parser.add_argument('-d', '--device', default='cuda', 
                    help='computation device to use', 
                    choices=['cpu', 'cuda'])
args = vars(parser.parse_args())
# Set the computation device.

###########
DEVICE = args['device']

def load_efficientnet_model():
    #Load the pre-trained EfficientNetB0 model.
    model = models.efficientnet_b0(pretrained=True)
    model.eval()
    return model

def load_googlenet_model():
    #Load the pre-trained googlenet model.
    model = models.googlenet(pretrained=True)
    model.eval()
    return model

def load_resnet18_model():
    #Load the pre-trained resnet18 model.
    model = models.resnet18(pretrained=True)
    model.eval()
    return model
    
def load_resnet50_model():
    #Load the pre-trained resnet50 model.
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

def load_resnet101_model():
    #Load the pre-trained resnet101 model.
    model = models.resnet101(pretrained=True)
    model.eval()
    return model

def load_vgg16_model():
    #Load the pre-trained vgg16 model.
    model = models.vgg16(pretrained=True)
    model.eval()
    return model

def load_vgg19_model():
    #Load the pre-trained vgg16 model.
    model = models.vgg19(pretrained=True)
    model.eval()
    return model

def preprocess():
    transform =  transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),])
    return transform


def read_classes():
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories


def inference(count, user_request, total, model_num, stopping, process_num, producer_status, finisher):
    pid = os.getpid()
    model_name = None
    print('Inference PID : {0}'.format(pid))
    if(model_num == 1):
        model = load_efficientnet_model()
        model_name = "efficient"
    elif(model_num == 2):
            model = load_googlenet_model()
            model_name = "googlenet"
    elif(model_num == 3):
            model = load_resnet18_model()
            model_name = "resnet18"
    elif(model_num == 4):
            model = load_resnet50_model()
            model_name = "resnet50"
    elif(model_num == 5):
            model = load_resnet101_model()
            model_name = "resnet101"
    elif(model_num == 6):
            model = load_vgg16_model()
            model_name = "vgg16   "
    elif(model_num == 7):
            model = load_vgg19_model()
            model_name = "vgg19   "

    # Initialize the model.
    model.to(DEVICE)
    # Load the ImageNet class names.
    categories = read_classes()
    # Initialize the image transforms.
    transform = preprocess()
    print(f"Computation device: {DEVICE}")

    image = cv2.imread(args['input'])
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Apply transforms to the input image.
    input_tensor = transform(rgb_image)
    # Add the batch dimension.
    input_batch = input_tensor.unsqueeze(0)
    # Move model to the computation device.  
    input_batch = input_batch.to(DEVICE)

    # warm up
    with torch.no_grad():
        for _ in range(100):
            output = model(input_batch)   

    # for simultaneous starting of inferences
    # giving continuous inference with while loop
    stopping.value = stopping.value + 1
    while(stopping.value < process_num):
        with torch.no_grad():
            output = model(input_batch)   
    
    print("{0} start!".format(model_name))
    starttime = time.time()
    processed_requests = 0
    with torch.no_grad():
        while True:
            if (user_request.empty()):
                if( producer_status.value == 0):        #user_request는 비어있지만 request creator는 아직 작동중
                    continue
                elif (producer_status.value == 1):      #user_request는 비어있고 request creator 작동 끝 --> break
                    break
            elif(user_request.qsize() > 0):
                file = open("result.txt","a")
                request_count = count.qsize()       #처리된 request가 X개를 넘으면 inference 종료
                if request_count > (total - 1):
                    print("PID: {0}, {1} inference process finish, {2} requests processed".format(pid, model_name, processed_requests))
                    file.close()
                    finisher.value = 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                    break
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                queueing_delay = time.time()-user_request.get()
                if queueing_delay < 0:
                    queueing_delay = 0
                start_time.record()
                output = model(input_batch)

                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top5_prob, top5_catid = torch.topk(probabilities, 1)    
                for i in range(top5_prob.size(0)):
                    print("{0}".format(processed_requests), categories[top5_catid[i]], top5_prob[i].item())

                end_time.record()
                torch.cuda.synchronize()

                processed_requests += 1
                pure_inference_time = (start_time.elapsed_time(end_time))*0.001     #change to milliseconds
                sum_time = queueing_delay + pure_inference_time
                file.write(f"{model_name} \t {(queueing_delay * 1000):.6f} \t {(pure_inference_time * 1000):.6f} \t {(sum_time * 1000):.6f}\n") 
                #convert results to milliseconds
                #print(f"Pure inference time : {(end_time-start_time):.6f} seconds")
                count.put(0)
                #print(user_request.qsize())
            file.close()
    endtime = time.time()
    print(f"total inference time : {(endtime-starttime):.6f} seconds")
            
#################################################################

# def input_lambda():
#     _lambda = float(input('Input lambda: '))
#     return _lambda

# def input_Interval():
#     Interval = float(input('Input Interval: '))
#     return Interval

# def distribute_poisson(_lambda, k):
#     """Return tuple of `n` values distributed using Poisson method."""
#     return ((_lambda) ** k) * exp(-_lambda) / factorial(k)

# def generate_value_poisson(_lambda):
#     random_value = random()
#     result = 0
#     probabilites_sum = distribute_poisson(_lambda, result)
#     while random_value > probabilites_sum:
#         result += 1
#         probabilites_sum += distribute_poisson(_lambda, result)
#     return result

#request_arrival은 expected value를 lambda로 하고, Interval과 Lambda를 직접 받는다. 
# user_request는 queue이다.
#
def request_arrival(user_request, _lambda, stopping, process_num, producer_status, finisher):
    pid = os.getpid()
    print("request producer PID: ", pid)

    not_processed = 0
    mod_lambda = _lambda * 1000                  #convert into milliseconds

    poisson_list = random.poisson(mod_lambda, 100000)       
    poisson_list = poisson_list / 1000          #convert to seconds

    time.sleep(2)                               #이거 안넣으면 queuing delay 값 이상하게 나옴.

    # for simultaneous starting of request creator with inference processes
    # inference process들이 inference할 준비가 되면 stopping.value가 process_num과 같아짐.
    # 이때부터 request를 생성하기 시작.

    while(stopping.value < process_num):
        continue

    print("request creator start!")     

    for timeslot in poisson_list:
        time.sleep (timeslot)
        requested_time = time.time()
        if user_request.qsize() < 50:                           
                user_request.put(requested_time)
        elif(finisher.value == 1):
            break                                           #inference 1000개 끝나면 request creater도 종료
        else:
            not_processed += 1
            continue                                            #쌓인 queue가 50개 이상이면 user_request에 넣지 않음
    producer_status.value = 1
    print(pid, ": request producer finish")
    print('{0} requests not added to request queue'.format(not_processed))

#############################################################

def get_total_requests():
    total = int(input('Input total requests to be processed: '))
    return total

def get_total_process():
    process_num = int(input('Input max processes: '))
    return process_num

# def get_lambda():
#     _lambda = int(input('Input Lambda: '))
#     return _lambda

# def get_interval():
#     Interval = int(input('Input Interval: '))
#     return Interval

# def choose_device():
#     which_device = int(input("what device? 1. RTX3090    2. GTX 1660ti   3. Jetson : "))
#     return which_device

# def busy_request():
#     is_busy = int(input("request frequency? 1. standard   2. 1/2    3. 1/4  : "))
#     return is_busy

# def choose_inteval():
#     which_interval = int(input("which interval? 1. smallest    2. average : "))
#     return which_interval


stopping = Value('i', 0)  #for simultaneous starting of inferences between processes
producer_status = Value('i', 0)     #if request producer is done, turn into 1
finisher = Value('i', 0)     #if request producer is done, turn into 1

if __name__ == '__main__':

    ###In GTX 1660ti
    interval_effi = 0.005235
    interval_googlenet = 0.004560
    interval_resnet18 = 0.002253
    interval_resnet50 = 0.005412
    interval_resnet101 = 0.009541
    interval_vgg16 = 0.009404
    interval_vgg19 = 0.011335
    
    #which_interval = choose_inteval()
    # which_device = choose_device()

    # if(which_interval == 1):                #arbitrary time
    #     if(which_device == 1):
    #         interval_effi = 0.0005
    #         interval_googlenet = 0.0005
    #         interval_resnet18 = 0.0005
    #         interval_resnet50 = 0.0005
    #         interval_resnet101 = 0.0005
    #         interval_vgg16 = 0.0005
    #         interval_vgg19 = 0.0005
    #     elif(which_device == 2):
    #         interval_effi = 0.00540280827617646
    #         interval_googlenet = 0.00474780042862891
    #         interval_resnet18 = 0.00224861196887492
    #         interval_resnet50 = 0.00552831678247451
    #         interval_resnet101 = 0.0100909974913597
    #         interval_vgg16 = 0.0093883776087761
    #         interval_vgg19 = 0.0113263149824142
    #     elif(which_device == 3):
    #         interval_effi = 0.0349936080532074
    #         interval_googlenet = 0.0297957528114319
    #         interval_resnet18 = 0.0125229693264961
    #         interval_resnet50 = 0.0290997981958389
    #         interval_resnet101 = 0.0540955404949188
    #         interval_vgg16 = 0.0440056154747009
    #         interval_vgg19 = 0.0537469785480498

    # elif(which_interval == 2):
    #     if(which_device == 1):
    #         interval_effi = 0.009115801
    #         interval_googlenet = 0.008309482
    #         interval_resnet18 = 0.00327271
    #         interval_resnet50 = 0.007339655
    #         interval_resnet101 = 0.0141879
    #         interval_vgg16 = 0.002875125
    #         interval_vgg19 = 0.003343757
    #     elif(which_device == 2):
    #         interval_effi = 0.00540280827617646
    #         interval_googlenet = 0.00474780042862891
    #         interval_resnet18 = 0.00224861196887492
    #         interval_resnet50 = 0.00552831678247451
    #         interval_resnet101 = 0.0100909974913597
    #         interval_vgg16 = 0.0093883776087761
    #         interval_vgg19 = 0.0113263149824142
    #     elif(which_device == 3):
    #         interval_effi = 0.0349936080532074
    #         interval_googlenet = 0.0297957528114319
    #         interval_resnet18 = 0.0125229693264961
    #         interval_resnet50 = 0.0290997981958389
    #         interval_resnet101 = 0.0540955404949188
    #         interval_vgg16 = 0.0440056154747009
    #         interval_vgg19 = 0.0537469785480498
    

    # is_busy = busy_request()

    # #if 2,  twice longer interval
    # if(is_busy == 2):
    #     interval_effi = interval_effi * 2
    #     interval_googlenet = interval_googlenet * 2
    #     interval_resnet18 = interval_resnet18 * 2
    #     interval_resnet50 = interval_resnet50 * 2
    #     interval_resnet101 = interval_resnet101 * 2
    #     interval_vgg16 = interval_vgg16 * 2
    #     interval_vgg19 = interval_vgg19 * 2

    # #if 3,  four times longer interval
    # elif(is_busy == 3):
    #     interval_effi = interval_effi * 4
    #     interval_googlenet = interval_googlenet * 4
    #     interval_resnet18 = interval_resnet18 * 4
    #     interval_resnet50 = interval_resnet50 * 4
    #     interval_resnet101 = interval_resnet101 * 4
    #     interval_vgg16 = interval_vgg16 * 4
    #     interval_vgg19 = interval_vgg19 * 4

    user_request = Queue()
    count = Queue()

    # start_time = time.time()
    total = get_total_requests()
    process_num = get_total_process()
    process_iter = process_num

    children = []
    # while (proc > 0):

    # minimum_inf = 0
    # maximum_inf = 0
    average_inf = 0
    interval_list = []

    while(process_iter > 0):
        print("Choose testing model. 1. EfficientNet, 2. GoogleNet, 3. ResNet18, 4. ResNet50")
        a = int(input("5. ResNet101, 6. VGG16, 7. VGG19 : "))
        if (a == 1):
            children.append(Process(target=inference, args=(count, user_request, total, 1, stopping, process_num, producer_status, finisher)))
            interval_list.append(interval_effi)
        elif (a == 2):
            children.append(Process(target=inference, args=(count, user_request, total, 2, stopping, process_num, producer_status, finisher)))
            interval_list.append(interval_googlenet)
        elif (a == 3):
            children.append(Process(target=inference, args=(count, user_request, total, 3, stopping, process_num, producer_status, finisher)))
            interval_list.append(interval_resnet18)
        elif (a == 4):
            children.append(Process(target=inference, args=(count, user_request, total, 4, stopping, process_num, producer_status, finisher)))
            interval_list.append(interval_resnet50)
        elif (a == 5):
            children.append(Process(target=inference, args=(count, user_request, total, 5, stopping, process_num, producer_status, finisher)))
            interval_list.append(interval_resnet101)
        elif (a == 6):
            children.append(Process(target=inference, args=(count, user_request, total, 6, stopping, process_num, producer_status, finisher)))
            interval_list.append(interval_vgg16)
        elif (a == 7):
            children.append(Process(target=inference, args=(count, user_request, total, 7, stopping, process_num, producer_status, finisher)))
            interval_list.append(interval_vgg19)

        process_iter = process_iter - 1
    
    #minimum_inf = min(interval_list)             #minimum inference time
    #maximum_inf  = max(interval_list)            #maximum inference time
    average_inf =st.mean(interval_list)


    p0 = Process(target=request_arrival, args=(user_request, average_inf, stopping, process_num, producer_status, finisher))
    # starttime = time.time()


    children.append(p0)             #request generator into children list
    for child in children:
        child.start()

    for child in children:
        child.join()

    # p0.start()
    # p1.start()
    # p2.start()
    # # p3.start()
    # # p4.start()

    # p0.join()
    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
    # endtime = time.time()
    print('total processed requests : {0}'.format(count.qsize()))
    # print("total time : {0}".format(endtime-starttime))



    # print(f"{(time.time() - start_time):.6f}")

# Get the softmax probabilities.


# probabilities = torch.nn.functional.softmax(output[0], dim=0)

# top5_prob, top5_catid = torch.topk(probabilities, 1)    

# for i in range(top5_prob.size(0)):
#     cv2.putText(image, f"{top5_prob[i].item()*100:.3f}%", (15, (i+1)*30), 
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.putText(image, f"{categories[top5_catid[i]]}", (160, (i+1)*30), 
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1, (0, 0, 255), 2, cv2.LINE_AA)
#     print(categories[top5_catid[i]], top5_prob[i].item())

# cv2.imshow('Result', image)
# cv2.waitKey(0)

# Define the outfile file name.
# save_name = f"outputs/{args['input'].split('/')[-1].split('.')[0]}_{DEVICE}.jpg"
# cv2.imwrite(save_name, image)