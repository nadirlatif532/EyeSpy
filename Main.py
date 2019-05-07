
import kivy
from kivy.app import App
#Configuration
from kivy.config import Config
Config.set('kivy','desktop','1')
#Config.set('graphics','borderless','0')
Config.set('graphics','position','custom')
Config.set('graphics','left',700)
Config.set('graphics','top',200)

import feature_extractor
from kivy.clock import Clock
from kivy.utils import rgba,get_color_from_hex
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.graphics import Color,Rectangle
from kivy.core.text import LabelBase
from kivy.uix.image import Image,AsyncImage
from kivy.properties import ObjectProperty,StringProperty
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition, FallOutTransition
from kivy.lang import Builder
from kivy.clock import Clock, mainthread
from kivy.uix.behaviors import DragBehavior
from kivy.uix.widget import Widget
import kivy.uix.videoplayer
from kivy.uix.videoplayer import VideoPlayer
from kivy.uix.video import Video
from kivy.uix.button import Button,ButtonBehavior
from kivy.garden.filebrowser import FileBrowser
from kivy.factory import Factory
from kivy.uix.popup import Popup
from kivy.utils import platform
import string
import os
import anomalydetector
from threading import Thread
from kivy.factory import Factory
from kivy.animation import Animation
from kivy.uix.scrollview import ScrollView
from kivy.uix.checkbox import CheckBox
import cv2
import shutil
import sqlite3
import torch
from functools import partial


'''
Global Declaration of Required Properties
'''
LabelBase.register(name = 'Helvetica', fn_regular='Helvetica_Regular.ttf', fn_bold='Helvetica_Bold.ttf')
Builder.load_file('eyespy_kv.kv')
path = './Appdata/Eyespy_noext.mp4'
feature_path = './Appdata/temp/textfeatures/'
Snippet_List = list()
dbName = "Appdata/eyespy.db"
table_name = "login"

class ImageButton(ButtonBehavior, AsyncImage):
    pass

class DisplayRoot(GridLayout):
    pass

class ScrollScreen(ScrollView):
    pass


class LoginScreen(Screen):
    '''
        Login Screen of the Application
    '''
    def __init__(self, **kwargs):
        super(LoginScreen, self).__init__(**kwargs)
        Window.size = (400, 600)

    def Login(self):
        '''
        Input: None
        Output: None
        Desc: Check Login info from DB and Proceed to MainMenu if Success
        else Show Error

        '''
        Username = self.ids.username.text
        Password = self.ids.password.text
        conn = sqlite3.connect(dbName)
        cursor = conn.execute("""SELECT username, password from login WHERE username=(?) and password=(?)""",
                              (Username, Password))
        record_exists = False
        for r in cursor:
            record_exists = True
            if Username == r[0] and Password == r[1]:
                print("Login Successful")
                self.ids.login_error.opacity = 0
                self.ids.loginbtn.color = rgba('#c8c8c8')
                Window.size = (1024, 768)
                Window.left = 500
                Window.top = 150
                Screen_Manager.current = 'MainMenu'

        if record_exists == False:
            self.ids.login_error.opacity = 1


    def on_pre_enter(self):
        Window.size = (400, 600)


    def changeColor(self):
        self.ids.loginbtn.color = rgba('#DAA520')

class MainMenu(Screen):
    '''
        MainMenu, Contains all the Anomaly Detection Frontend (Offline)
    '''
    def __init__(self,**kwargs):
        '''
        Desc: Define the MainMenu Class Variables

        '''
        global path
        super(MainMenu, self).__init__(**kwargs)
        self.file_popup = FilePopup()
        self.SnippetList = list()
        self.GPU_Flag = False
        self.Batch_Flag = False
        self.SS = ScrollScreen()
        self.popup = Popup(title='Please Wait',
                      content=Label(text='Video is being processed', color=rgba('#DAA520'), font_size=24),
                      auto_dismiss=False, size_hint=(0.5, 0.5))

    def on_pre_enter(self):
        '''
        Desc: Test if a CUDA enables GPU is present. If not Disable the CPU/GPU Switch

        '''
        settings = App.get_running_app().root.get_screen("Settings")
        if torch.cuda.device_count() > 0:
            self.GPU_Flag = True
            settings.ids.check.active = True
        else:
            settings.ids.check.disabled = True
            settings.ids.check.opacity = 0
            settings.ids.gpu_text.opacity = 0

        Window.borderless = False
        #Window.fullscreen = 'auto'
        Window.position = 'custom'

    def on_enter(self, *args):
        '''
        Desc: AutoPlay the Video on Enter
        '''
        self.ids.videoplayer.state = 'play'

    def Set_Gpu(self,state):
        '''
        Input: State of GPU Checkbox in Settings

        Desc: Set GPU Flag to control GPU Execution
        '''

        self.GPU_Flag = state
        print("Executing on GPU: " + str(self.GPU_Flag))

    def filebrowse(self):
        '''
        Desc: Called When Add Video Button is Pressed, works:
        1) Clear Widget Scrollscreen
        2) Clear Temp folder
        3) Create New Temp Directories
        4) Open FileBrowser

        '''

        self.SnippetList = []
        self.ids.plot_image.opacity = 0
        self.ids.Snippets.remove_widget(self.SS)
        self.SS = ScrollScreen()
        self.ids.videoplayer.source = ''
        if os.path.exists('./Appdata/temp'):
            shutil.rmtree("./Appdata/temp/")

        os.makedirs('./Appdata/temp')
        os.makedirs('./Appdata/temp/snip')
        os.makedirs('./Appdata/temp/frames')
        os.makedirs('./Appdata/temp/textfeatures')
        os.makedirs('./Appdata/temp/plot')
        self.popup.content = Label(text='Features are being extracted..(1/3)', color=rgba('#DAA520'), font_size=24)
        self.file_popup.start()

    def change_to_live(self):
        Screen_Manager.current = 'Live'

    def image_press(*args):
        Screen_Manager.current = 'Settings'

    def changevideo(self):
        '''
        desc: Change VideoPlayer Source when a snippet is clicked.

        '''
        global path
        mainmenu = App.get_running_app().root.get_screen('MainMenu')
        mainmenu.ids.videoplayer.source = ''
        mainmenu.ids.videoplayer.source = path
        print(path)
        return



    def batch_processing(self):

        self.ids.videoplayer.state = 'pause'

        self.Batch_Flag = True
        self.popup.content = Label(
                text='Videos are being processed...This will take long',
                color=rgba('#DAA520'), font_size=20)
        self.popup.open()
        self.featureExtraction()


    def featureExtraction(self):

        '''
        Desc: Called when the Videoplayer source is changed. Creates a thread and calls the feature extractor)

        '''
        global path
        mainmenu = App.get_running_app().root.get_screen('MainMenu')
        if self.Batch_Flag == False:

            if mainmenu.ids.videoplayer.source == '':
                return
            if '_noext' not in mainmenu.ids.videoplayer.source:
                thread = Thread(target=feature_extractor.feature_extractor, args = (feature_path, path, './Appdata/temp/snip/', 6, mainmenu.GPU_Flag))
                thread.daemon = True
                thread.start()
                self.popup.open()
            else:
                pass
        else:
            try:
                f = open('./Appdata/config.txt')
                lines = f.readlines()
                f.close()
                video_input_path = lines[0]
                video_input_path = video_input_path[:-1]
                input_path = video_input_path
                full_path = input_path

                input_path = os.listdir(input_path)
            except:
                print("Please set the input directory")
            video_count = 0
            videos_in_folder = []

            for videos in input_path:
                if videos.endswith('.mp4'):
                    video_count = video_count+1
                    videos_in_folder.append(videos)

            thread = Thread(target=self.Batch_Loop,
                            args=(feature_path, full_path, './Appdata/temp/snip/', 6, mainmenu.GPU_Flag,videos_in_folder, video_count))
            thread.daemon = True
            thread.start()



    def Batch_Loop(self, feature_path, full_path, snippet_path, batch_size, gpu_flag,videos_in_folder, video_count ):

        current_video = 0
        self.popup.content = Label(
            text='Processing Video..(' + str(current_video)+ '/' + str(video_count) + ')' ,
            color=rgba('#DAA520'), font_size=20)
        for video in videos_in_folder:
            if '_noext' not in video:
                current_video = current_video + 1
                self.popup.content = Label(
                    text='Processing Video..(' + str(current_video) + '/' + str(video_count) + ')',
                    color=rgba('#DAA520'), font_size=20)
                path = full_path + '/' + video
                feature_extractor.feature_extractor(feature_path,path,snippet_path,batch_size,gpu_flag)

        self.popup.content = Label(
            text='Anomalous Snippets of Videos saved in the output folder',
            color=rgba('#DAA520'), font_size=20)
        Clock.schedule_once(partial(self.dismisspopup), 1)
        mainmenu = App.get_running_app().root.get_screen('MainMenu')
        mainmenu.ids.videoplayer.state = 'play'




    def SaveSnippet(self):

        '''
        Desc: Called when the save snippet button is clicked.
        1) If no Snippets are selected, Show popup and wait for selection.

        else

        2) concatenate the snippets and save them in the output directory
        3) clear the snippet lists and populate it with the new snippets

        '''

        self.popup.content = Label(text='Saving Snippets', color=rgba('#DAA520'), font_size=24)
        self.ids.videoplayer.state = 'stop'
        try:
            f = open('./Appdata/config.txt')
            lines = f.readlines()
            f.close()
            video_output_path = lines[1]
            video_output_path = video_output_path[:-1]
            output_path = video_output_path
        except:
            output_path = 'Appdata/output/'
        vidname = os.path.basename(path)
        list_path = 'Appdata/temp/snip/'

        if not self.SnippetList:
            self.popup.content = Label(text='Please add Snippets to save', color=rgba('#DAA520'), font_size=24)
            self.popup.open()
            Clock.schedule_once(partial(self.dismisspopup),1)

        else:
            self.popup.open()
            with open('Appdata/temp/snip/snippets.txt', 'w') as file:
                for element in self.SnippetList:
                    file.writelines('file ' + '\'' + element + "\'\n")
                file.close()
            os.system("ffmpeg -f concat -i {}snippets.txt -codec copy {}/{}_anomalous.mp4".format(list_path, output_path,

                                                                                                  vidname))
            self.dismisspopup()
            self.SnippetList = []
            self.ids.plot_image.opacity = 0
            self.ids.Snippets.remove_widget(self.SS)
            self.SS = ScrollScreen()
            self.ids.videoplayer.source = '.\\Appdata\\Eyespy_noext.mp4'
            self.ids.videoplayer.state = 'play'

    def dismisspopup(self, *args):

        self.popup.dismiss()







class Snippet(GridLayout):

    def __init__(self, *args, **kwargs):
        '''
        Desc: Create a Snippet object from *args

        '''
        super(Snippet, self).__init__(**kwargs)
        self.ids.thumbnail.source = args[0]
        self.ids.severity.source = args[1]
        self.ids.check.group = args[2]

    def thumb_to_video(self,thumb_source):
        '''
        Desc: Change VideoPlayer Source when a snippet is clicked.
        '''
        thumb_source = thumb_source[:-8]
        vid_path = thumb_source + '_noext' + '.mp4'

        mainmenu = App.get_running_app().root.get_screen("MainMenu")
        mainmenu.ids.videoplayer.source = vid_path
        mainmenu.ids.videoplayer.state = 'play'

    def add_snippet(self,group,value):
        '''
        Desc: toggle the Snippet in the SnippetList
        '''
        mainmenu = App.get_running_app().root.get_screen("MainMenu")

        if value == True:
            mainmenu.SnippetList.append(group)
        else:
            mainmenu.SnippetList.remove(group)
        #print(mainmenu.SnippetList)





class Live(Screen):
    '''
        Live Screen (Implementation In progress)
    '''
    def __init__(self, **kwargs):
        super(Live, self).__init__(**kwargs)

    def on_pre_enter(self):
        # Window.fullscreen = 'auto'
        Window.position = 'custom'


    def image_press(*args):
        Screen_Manager.current = 'Settings'

    def change_to_offline(self):
        Screen_Manager.current = 'MainMenu'

class Settings(Screen):
    '''
        Settings screen for changing options
    '''
    def __init__(self, **kwargs):
        '''
        Desc: Read config file and show paths

        '''
        super(Settings, self).__init__(**kwargs)
        self.file_popup = FilePopup()
        self.flag = 0

        try:
            f = open('./Appdata/config.txt')
            lines = f.readlines()
            video_input_path = lines[0]
            video_input_path = video_input_path[:-1]
            f.close()
            video_output_path = lines[1]
            video_output_path = video_output_path[:-1]
            self.ids.inputvideopath.text = video_input_path
            self.ids.outputvideopath.text = video_output_path
        except:
            print("Config exception")


    def filebrowse_input(self):
        '''
        Desc: Set input folder for videos

        '''
        self.flag = 0
        self.file_popup.start()

    def filebrowse_output(self):
        '''
            Desc: Set output folder for videos

        '''
        self.flag = 1
        self.file_popup.start()

    def load_paths(self):
        '''
        Desc: Reload paths from config file

        '''
        try:
            f = open('./Appdata/config.txt')
            lines = f.readlines()
            video_input_path = lines[0]
            video_input_path = video_input_path[:-1]
            f.close()
            video_output_path = lines[1]
            video_output_path = video_output_path[:-1]
            self.ids.inputvideopath.text = video_input_path
            self.ids.outputvideopath.text = video_output_path
        except:
            print("Config exception")


    def change_to_live(self):
        Screen_Manager.current = 'Live'

    def change_to_offline(self):
        Screen_Manager.current = 'MainMenu'

    def Set_GPU(self):
        '''
        Desc: Change state of GPU_flag

        '''
        mainmenu = App.get_running_app().root.get_screen("MainMenu")

        if self.ids.check.active == True:
            mainmenu.Set_Gpu(True)
        else:
            mainmenu.Set_Gpu(False)

class FilePopup():
    '''
        Desc: The Filebrowser class
    '''
    def __init__(self, short_text='File Browser'):

        try:
            f = open('./Appdata/config.txt')
            lines = f.readlines()
            video_input_path = lines[0]
            video_input_path = video_input_path[:-1]
            video_input_path = os.path.abspath(video_input_path)
            video_output_path = lines[1]
            video_output_path = video_output_path[:-1]
            video_output_path = os.path.abspath(video_output_path)
            f.close()

        except:
            video_input_path = './Appdata/InputVideos/'
            video_output_path = './Appdata/output/'

        browser = FileBrowser(select_string='Select', cancel_state='down',favorites = [(video_input_path,"Input Folder"),(video_output_path,"Output Folder")])
        browser.bind(on_success=self._fbrowser_success,
                     on_canceled=self._fbrowser_canceled,
                    )

        self.popup = Popup(
            title=short_text,
            content=browser, size_hint=(0.7, 0.7),
            auto_dismiss=False,
            background_color = rgba('#2f323a'),
            separator_color = rgba('#DAA520'),
            title_color = rgba('#DAA520')
        )

    def _fbrowser_canceled(self, instance):
        '''
        Desc: File Browser cancel event, Do nothing and close file browser

        '''
        self.popup.dismiss()

    def _fbrowser_success(self, instance):
        '''
            Desc: File Browser Success event, sets the correct path

        '''
        global path
        path = instance.selection
        path = ''.join(path)
        if os.path.isfile(path):
            if path.endswith('.mp4'):
                mainmenu = App.get_running_app().root.get_screen('MainMenu')
                mainmenu.changevideo()
                self.popup.dismiss()
            else:
                self.popup.dismiss()
        if os.path.isdir(path):
            settings = App.get_running_app().root.get_screen('Settings')
            flag = settings.flag

            f = open('./Appdata/config.txt')
            lines = f.readlines()
            if flag == 0:
                lines[0] = path + '\n'
            else:
                lines[1] = path + '\n'
            f.close()
            f = open('./Appdata/config.txt', 'w')
            f.writelines(lines)
            f.close()
            settings.load_paths()
            self.popup.dismiss()


    def start(self):
        self.popup.open()


'''
    Desc: ScreenManager to define the structure of the application 
'''
Screen_Manager = ScreenManager(transition = FadeTransition())

Screen_Manager.add_widget(LoginScreen(name = "LoginScreen"))
Screen_Manager.add_widget(MainMenu(name = "MainMenu"))
Screen_Manager.add_widget(Live(name = "Live"))
Screen_Manager.add_widget(Settings(name = "Settings"))



class EyeSpy(App):

    '''
    Desc: Main App Class, Entry point of the application
    '''
    def build(self):
        '''
        Desc: The entry function of the application:
        1) Clear the temp directory.
        2) Create config if it doesn't exist
        3) Create Connection with the Database
        4) Return ScreenManager as the Root Widget


        '''

        if not os.path.exists('./Appdata'):
            os.makedirs('./Appdata')
            os.makedirs('./Appdata/InputVideos')
            os.makedirs('./Appdata/output')
        if os.path.exists('./Appdata/temp'):
            shutil.rmtree("./Appdata/temp/")

        os.makedirs('./Appdata/temp')
        os.makedirs('./Appdata/temp/snip')
        os.makedirs('./Appdata/temp/frames')
        os.makedirs('./Appdata/temp/textfeatures')
        os.makedirs('./Appdata/temp/plot')

        if not os.path.exists('./Appdata/config.txt'):
            path_seq = ['./Appdata/InputVideos\n','./Appdata/output\n','./Appdata/temp/frames\n','./Appdata/temp/snip\n']
            f = open("./Appdata/config.txt", "w+")
            f.writelines(path_seq)
            f.close()
        if not os.path.exists(dbName):
            try:
                conn = sqlite3.connect(dbName)
                print("DB {} connection success".format(dbName))
            except:
                print("Connection failed with the database")

            table_name = "login"
            sql = "create table if not exists " + table_name + " (username string, password string, admin integer)"
            conn.execute(sql)
            conn.commit()
            conn.execute("""insert into login (username, password,admin) VALUES (?, ?, ?);""", ('admin', 'admin','1'))
            conn.commit()

        Window.borderless = True
        self.icon = 'Media/eyespy_notext.png'
        return Screen_Manager



if __name__ == '__main__':
    EyeSpy().run()