Completes the Lunar lander environment by Open AI, using a reinforcement Double Deep Q-learning Network.

###Installation
```
pip install requirements.txt
```
* The Lunar Lander module from Open AI gym's module requires some additional libraries to function propperly,  on windows the current described version of Box2D works well, but for other platforms different libraries may need to be used. If the above fails to install correctly please refer to  [here](https://gym.openai.com/docs/#:~:text=Download%20and%20install%20using%3A,full%20installation%20containing%20all%20environments) for how to install on your particular platform.

###Training Script
```
python3 train.py --episodes 2000 --val True -- early_stop 200
```

###Validation Script
```
python3 validate.py --episodes 100 --checkpoint "SavedModels/LunarLanderModel.pt"
```

