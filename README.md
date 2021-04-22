# Three_Class_Mask_Detector

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
  
