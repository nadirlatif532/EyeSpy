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
from kivy.uix.filechooser import FileChooserIconView
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

LabelBase.register(name = 'Helvetica', fn_regular='Helvetica_Regular.ttf', fn_bold='Helvetica_Bold.ttf')
Builder.load_file('eyespy_kv.kv')
path = '.\\Appdata\\Eyespy.mp4'
feature_path = './Appdata/temp/textfeatures/'
Snippet_List = list()
popup = Popup(title='Please Wait', content=Label(text='Video is being processed', color = rgba('#DAA520'), font_size = 24, underline = True),
                          auto_dismiss=False,size_hint = (0.5,0.5))
dbName = "Appdata/eyespy.db"
table_name = "login"
class LoginScreen(Screen):

    def __init__(self, **kwargs):
        super(LoginScreen, self).__init__(**kwargs)
        Window.size = (400, 600)

    def Login(self):
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
                Screen_Manager.current = 'MainMenu'
                Window.size = (1024, 768)
                Window.left = 500
                Window.top = 150
        if record_exists == False:
            self.ids.login_error.opacity = 1


    def on_pre_enter(self):
        Window.size = (400, 600)


    def changeColor(self):
        self.ids.loginbtn.color = rgba('#DAA520')

class MainMenu(Screen):

        def __init__(self,**kwargs):
            global path
            super(MainMenu, self).__init__(**kwargs)
            self.file_popup = FilePopup()
            self.SnippetList = list()
            self.GPU_Flag = True
            self.SS = ScrollScreen()

        def on_pre_enter(self):
            Window.borderless = False
            #Window.fullscreen = 'auto'
            Window.position = 'custom'
        def on_enter(self, *args):
            self.ids.videoplayer.state = 'play'

        def Set_Gpu(self,state):
            self.GPU_Flag = state
            print(self.GPU_Flag)

        def filebrowse(self):
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
            self.file_popup.start()
        def change_to_live(self):
            Screen_Manager.current = 'Live'

        def image_press(*args):
            Screen_Manager.current = 'Settings'

        def changevideo(self):
            global path
            mainmenu = App.get_running_app().root.get_screen('MainMenu')
            mainmenu.ids.videoplayer.source = ''
            mainmenu.ids.videoplayer.source = path
            print(path)
            return

        def featureExtraction(self):

            mainmenu = App.get_running_app().root.get_screen('MainMenu')
            if mainmenu.ids.videoplayer.source == '':
                return
            if '_noext' not in mainmenu.ids.videoplayer.source:
                thread = Thread(target=feature_extractor.feature_extractor, args = (feature_path, path, './Appdata/temp/snip/', 6, mainmenu.GPU_Flag))
                thread.daemon = True
                thread.start()
                popup.open()
            else:
                pass
        def SaveSnippet(self):
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
                print("Add Snippets first")
            else:

                with open('Appdata/temp/snip/snippets.txt', 'w') as file:
                    for element in self.SnippetList:
                        file.writelines('file ' + '\'' + element + "\'\n")
                    file.close()
                os.system("ffmpeg -f concat -i {}snippets.txt -codec copy {}/{}_anomalous.mp4".format(list_path, output_path,
                                                                                                      vidname))
            self.SnippetList = []
            self.ids.plot_image.opacity = 0
            self.ids.Snippets.remove_widget(self.SS)
            self.SS = ScrollScreen()
        def dismisspopup(self):

            popup.dismiss()


        def errormessage(self):
            print('Sahi File de bharwe')


class ImageButton(ButtonBehavior, AsyncImage):
    pass
class Snippet(GridLayout):
    def __init__(self, *args, **kwargs):
        super(Snippet, self).__init__(**kwargs)
        self.ids.thumbnail.source = args[0]
        self.ids.severity.source = args[1]
        self.ids.check.group = args[2]

    def thumb_to_video(self,thumb_source):
        thumb_source = thumb_source[:-8]
        vid_path = thumb_source + '_noext' + '.mp4'

        mainmenu = App.get_running_app().root.get_screen("MainMenu")
        mainmenu.ids.videoplayer.source = vid_path
        mainmenu.ids.videoplayer.state = 'play'

    def add_snippet(self,group,value):
        mainmenu = App.get_running_app().root.get_screen("MainMenu")

        if value == True:
            mainmenu.SnippetList.append(group)
        else:
            mainmenu.SnippetList.remove(group)
        print(mainmenu.SnippetList)
class DisplayRoot(GridLayout):
    pass

class ScrollScreen(ScrollView):
    pass


class Live(Screen):
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
    def __init__(self, **kwargs):
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
        self.flag = 0
        self.file_popup.start()
    def filebrowse_output(self):
        self.flag = 1
        self.file_popup.start()
    def load_paths(self):
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
        mainmenu = App.get_running_app().root.get_screen("MainMenu")
        if self.ids.check.active == True:
            mainmenu.Set_Gpu(True)
        else:
            mainmenu.Set_Gpu(False)

class FilePopup():
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
        print ('cancelled, Close self.')

        self.popup.dismiss()

    def _fbrowser_success(self, instance):
        global path
        path = instance.selection
        path = ''.join(path)
        #print(path)
        if os.path.isfile(path):
            if path.find('.mp4'):
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



Screen_Manager = ScreenManager(transition = FadeTransition())

Screen_Manager.add_widget(LoginScreen(name = "LoginScreen"))
Screen_Manager.add_widget(MainMenu(name = "MainMenu"))
Screen_Manager.add_widget(Live(name = "Live"))
Screen_Manager.add_widget(Settings(name = "Settings"))




class EyeSpy(App):

    def build(self):

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