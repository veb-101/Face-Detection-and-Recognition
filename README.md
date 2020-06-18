# Face Detection and Recognition using Haar-cascade

------

## Usage (python3)

1) Create a new folder names: **project_face_recog**. Copy all files from the repository to the created folder
2) Open cmd in the folder.  
3) Install dependencies

```python
pip install -r requirements.txt
```

4) Test whether opencv can detect camera.

```python
python demo_cam_test.py
```

5) Make sure the _.xml_ is present in the directory, check if face detection is working:

```python
python demo_face_detection.py
```

6) Start building dataset for training purposes.
7) Create a **dataset** folder in the same directory as the script.
8) To build a dataset for your face for recognition purposes.


```python
python face_dataset.py
```

* Inputs: face_id (int).
* Program will ask for an input.

```python
Enter user id end press <return> ==> 1
```

* Can repeat this step for multiple faces, just give a different id (sequential)
* This will take 30 images of your face and save them in the **dataset** folder
* At the end, the datasets folder will be opened, delete _bad_ images.

9) Create a **trainer** folder in the same directory as script.
10)  After dataset has been created, train a classifier on your face images.

* First update **line 16**
 
 ```python
 names = ['None'] # add your name corresponding to the id number like, 
 # names = ['None', 'Vaibhav', ...]
 ```

```python
python face_training.py
```


11)  After training, for recognition demo, execute:

```python
python face_recognition.py
```

## Main Reference:
[Real-Time Face Recognition: An End-To-End Project](https://towardsdatascience.com/real-time-face-recognition-an-end-to-end-project-b738bb0f7348)
