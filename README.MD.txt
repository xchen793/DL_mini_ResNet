#Statement
In this mini-project, we implemented and combined the model and train code into project1_model.py. The trained parameters are all saved in project1_model.py.

# How to run the code
* You can either download the project1_model.py and run it via VScode or terminal. If run by terminal, go to the folder where the res_train.py is downloaded and then use the command line here: python './res_train.py' 
* Since we have proposed a data augmentation method, which is the random erasing. If you want to add the random erasing(0.5,0.4,0.3), then the run by this command line: python './project1_model.py' --p 0.5 --sh 0.4 --r1 0.3. However, we did not take this method into our final model.
* If you want to use the checkpoint project1_model.pt, please uncommet the line 33, 61, 62 (where you shouuld see we load the PATH there), and comment the line 58 (net = resnet18().to(device)).
*If you have trouble running, please do not hesitate to contact me at xc2425@nyu.edu.
