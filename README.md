# Applied Deep Learning project : Autonomous_Vehicle

### Project Organization

------------
    │
    ├── Archive
    |    |-- This File contains a lot of models that we tried but that did not work. It also contains dated notebooks.
    |
    ├── data_collection_remote_control.ipynb
    │    |-- This notebook lays out functionality to control the GoPiGo using python(on raspian for robots).
    |
    ├── data_collection_capture.ipynb
    |    |--This notebook captures images while the car is driving, run after running data_collection_remote_control.ipynb.
    |
    ├──  model_training_final.ipynb
    |    |--This notebook contains code to train the model that we will finally use. It also preprocesses images.
    |
    ├── lane_keep_testing_final.ipynb
    |    |--This notebook uses a pretrained model( .h5 and .json) files to make online predictions on the GoPiGo.
    |
    ├── final_trained_model.h5, final_trained_model.json
    |    |--Trained Models
       
    
--------

## Final Github Files to keep track of:

Lane Keep
- data_collection_remote_control.ipynb - done
- data_collection_capture.ipynb - done
- wheel_speeds.csv (not on Git yet)
- training_images.tar.gz (not on Git yet)
- Final_model_training.ipynb - done
- final_trained_model.h5 done
- final_trained_model.json done
- lane_keep_testing_final.ipynb done 

Sign Detection
- ADL_project_Sign_ Recognition.ipynb
- object_detection.py
- heatmap.py
