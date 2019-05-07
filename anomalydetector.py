import os,shutil; os.environ['KERAS_BACKEND'] = 'theano'
from scipy.io import loadmat, savemat
from keras.models import model_from_json
from math import factorial
import numpy as np
import numpy
from numpy import  matlib
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import cv2
import os, sys
import Main
from kivy.uix.image import AsyncImage
from kivy.uix.label import Label
from kivy.utils import rgba
params = {"ytick.color" : "DAA520",
          "xtick.color" : "DAA520",
          "axes.labelcolor" : "DAA520",
          "axes.edgecolor" : "DAA520"}
plt.rcParams.update(params)


def load_model(json_path):
    model = model_from_json(open(json_path).read())
    return model

def load_weights(model, weight_path):
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

def conv_dict(dict2): # Helper function to save the model
    i = 0
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    #try:
    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))


    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")

    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)

    half_window = (window_size - 1) // 2
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y,mode='valid')








def load_dataset_One_Video_Features(Feature_Path):

    f = open(Feature_Path, "r")
    words = f.read().split()
    num_feat = len(words) / 4096
    # Number of features per video to be loaded. In our case num_feat=32, as we divide the video into 32 segments. Npte that
    # we have already computed C3D features for the whole video and divide the video features into 32 segments.

    count = -1;
    VideoFeatues = []
    for feat in range(0, int(num_feat)):
        feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
        count = count + 1
        if count == 0:
            VideoFeatues = feat_row1
        if count > 0:
            VideoFeatues = np.vstack((VideoFeatues, feat_row1))
    AllFeatures = VideoFeatues

    return  AllFeatures





def anomalydetector(vidpath,featpath,video_name):

    Model_dir = os.path.dirname(os.path.realpath(__file__))
    weights_path = os.path.join(Model_dir, 'weights_L1L2.mat')
    model_path = os.path.join(Model_dir, 'model.json')

    mainmenu = Main.App.get_running_app().root.get_screen("MainMenu")
    if mainmenu.Batch_Flag == False:
        mainmenu.popup.content = Label(text='Predicting Anomalies..(2/3)', color=rgba('#DAA520'), font_size=24)



    model = load_model(model_path)
    load_weights(model, weights_path)
    video_path = vidpath

    cap = cv2.VideoCapture(video_path)

    Total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_segments = np.linspace(1, Total_frames, num=33)
    total_segments = total_segments.round()

    feature_text = video_name + '.txt'

    FeaturePath = os.path.join(featpath, feature_text)


    inputs = load_dataset_One_Video_Features(FeaturePath)
    predictions = model.predict_on_batch(inputs)

    Frames_Score = []
    count = -1;
    for iv in range(0, 29):
        F_Score = np.matlib.repmat(predictions[iv],1,(int(total_segments[iv+1])-int(total_segments[iv])))
        count = count + 1
        if count == 0:
            Frames_Score = F_Score
        if count > 0:
            Frames_Score = np.hstack((Frames_Score, F_Score))




    cap = cv2.VideoCapture((video_path))
    while not cap.isOpened():
        cap = cv2.VideoCapture((video_path))
        cv2.waitKey(1000)
        print ("Wait for the header")

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    Total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(3))
    height = int(cap.get(4))


    print ("Anomaly Prediction")
    x = np.linspace(1, int(Frames_Score.size), int(Frames_Score.size))
    scores = Frames_Score
    scoressmoothed=scores.reshape((scores.shape[1],))
    scoressmoothed = savitzky_golay(scoressmoothed, 101, 3)
    plt.close()
    break_pt=min(scoressmoothed.shape[0], x.shape[0])
    plt.axis([0, Total_frames, 0, 1])
    i=j=k=0



    snip_n=1
    OUTPUT_DIR = os.path.dirname(os.path.realpath(__file__))
    AppData = os.path.join(OUTPUT_DIR, 'AppData')
    if not os.path.exists(AppData):
        os.makedirs(AppData)

    if mainmenu.Batch_Flag == False:
        mainmenu.popup.content = Label(text='Extracting Snippets..(3/3)', color=rgba('#DAA520'), font_size=24)
    framedir = os.path.join(AppData,os.path.join('temp','frames'))
    print(framedir)
    snipdir = os.path.join(AppData,os.path.join('temp','snip'))
    if not os.path.exists(snipdir):
        os.makedirs(snipdir)
    if (os.path.exists(framedir)):
        shutil.rmtree(framedir)
    if not os.path.exists(framedir):
        os.makedirs(framedir)

    snipdesc=[]
    sniplist=[]
    totalscore=0
    while True:
        frameId = cap.get(1)
        flag, frame = cap.read()
        threshold = np.mean(scoressmoothed)
        if (threshold < 0.0002):
            break

        framescore = scoressmoothed[int(frameId)]
        if (framescore > threshold):
            totalscore=totalscore+framescore
            cv2.imwrite('{}/{}.jpg'.format(framedir,int(frameId)),frame)
            if j==0:
                startframe=frameId
            j+=1
            k = int(frameId)

        elif (framescore <= threshold and int(frameId)-k>64 and j!=0):
            cv2.imwrite('{}/snip{}_pic.png'.format(snipdir, snip_n), frame)
            os.system(
                'ffmpeg -f image2 -framerate 30 -s {}x{} -start_number {} -i {}/%d.jpg -vcodec libx264 -profile:v high444 -refs 8 -crf 25 -preset ultrafast -pix_fmt yuv420p {}/snip{}_noext.mp4'.format(
                    width, height, startframe, framedir, snipdir, snip_n))
            snipdesc.append('snip{}_noext.mp4'.format(snip_n))
            snipdesc.append('snip{}_pic.png'.format(snip_n))
            totalscore = totalscore/Total_frames
            snipdesc.append(float("{0:.2f}".format(totalscore)))
            snipdesc.append(int(startframe))
            snipdesc.append(k)
            sniplist.append(np.array(snipdesc))
            snipdesc.clear()
            j=0
            totalscore=0
            snip_n += 1


        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES)== break_pt:
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break

    plotdir = os.path.join(AppData, os.path.join('temp', 'plot'))
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    if mainmenu.Batch_Flag == False:
        plotpath = os.path.join(plotdir,video_name[:-4])
    else:
        try:
            f = open('./Appdata/config.txt')
            lines = f.readlines()
            f.close()
            video_output_path = lines[1]
            video_output_path = video_output_path[:-1]
            output_path = video_output_path
        except:
            output_path = 'Appdata/output/'
        plotpath = os.path.join(output_path, video_name[:-4])
    fig = plt.figure()
    fig.patch.set_alpha(0)

    plot = fig.add_subplot(1, 1, 1)
    plot.set_facecolor('#545F66')

    plt.plot(x, scoressmoothed, color='#CC0000', linewidth=2)
    plt.savefig(plotpath)
    plt.close()


    if mainmenu.Batch_Flag == False:
        mainmenu.ids.plot_image.source = plotpath + '.png'
        mainmenu.ids.plot_image.opacity = 1
        mainmenu.dismisspopup()

    sniplist=(np.array(sniplist))


    return sniplist




