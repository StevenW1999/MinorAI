# MinorAI

Make sure to have the following modules installed using pip install
- sklearn
- seaborn
- pyplot
- spotipy (pip install spotipy --upgrade)
- opencv
- cmake
- dlib
- Pillow
- pandas
- NumPy

In order to use the live emotion recognition, make sure to have a webcam
If not, there is a test image in LiveCapture folder to use as example.
To use the recognition, 
1.  make sure to have a spotify account
2.  create a python file aan called spotipyconfig.py
- add the following code:
```
client_ID='1e8461af4b67428cb73810f0cf5ecbec
client_SECRET='c349975517a747ee9a952f6c31a156b5
redirect_url='https://example.com
```
2. run sp_main.py
3. press space to take a picture when the camera is open
4. make sure there is a blue box around your face when taking the picture
5. after taking the picture, the video capture will close.
6. you will be redirected to a page 
7. log in with your spotify and give access to the application
8. you will be redirected again
9. copy the url of the website to prove that you are the person giving permission
10. paste it in the console as specified
11. you will see a list of genres in the console
12. type in max 5 genres in the console
13. a new playlist will be made with the recommendations based on your emotion and genre
14. enjoy the music!

The project also consists of jupyter notebook files containing the algorithms and results of those algorithms.
simply run the jupyter notebook files to see the results.
There is also a All.ipynb file. This contains all algorithms in a single file.
NOTE: the KNN takes long to test so please do not worry if the algorithm takes a long time.
