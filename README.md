# Three_Class_Mask_Detector

Overview
-------------

https://user-images.githubusercontent.com/48312457/115974399-7d8d7780-a597-11eb-8308-5266389bbedd.mp4



This is a real-time mask search program using CNN. This can distinguish whether to wear a mask or not with 3 different labels. If you don't wear it properly and 10 seconds later, you'll get an image of wearing a mask properly. If you wear the mask properly after 10 seconds, the window will turn off. When I first decided on the topic, I thought about monitoring people's wearing masks in PC rooms.

And I has developed it by referring to many existing mask search algorithms, so please check the reference below. This is my first ML project.

Command
-------------

This is how to run
~~~
$ python3 detect_mask_video.py 
~~~
datasets
----------

These zip files are uploaded using LFS.

you should install git LFS.

ML
-------------
In colab,
~~~
!git clone https://github.com/tjddnghkrk/graduate.git
~~~
~~~
%cd ./graduate
!pip3 install -r requirements.txt
~~~
~~~
%cd datase
!git clone https://github.com/git-lfs/git-lfs.gi
!sudo apt-get install git-lf
!git lfs pull
~~~
~~~
%cd ./dataset
~~~ 
~~~
!unzip "with_mask.zip"
!unzip "without_mask.zip"
!unzip "wrong_mask.zip"
~~~
~~~
!rm -rf __MACOSX
!rm -rf git-lfs
!rm *.zip
~~~
~~~
%cd ../
~~~
~~~
!python3 train_mask.py --dataset dataset
~~~

Colab doesn't support webcams, so download the learned model and run detect_mask_video.py locally.



Reference
---------
https://github.com/chandrikadeb7/Face-Mask-Detection

https://leimao.github.io/blog/Git-Large-File-Storage/

https://arxiv.org/abs/2008.08016


  
