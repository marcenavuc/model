# model
#Ctrl + C Ctrl + V

```python
from keras.models import model_from_json
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("kaggle_unet.hdf5")
print("Loaded model from disk")
```

# Если что-то идет не так, то
1) Скопируй датасет 
```!git clone https://github.com/marcenavuc/dataset_capseq.git```
1) Импортируй utils
2) Примерный код дата генераторов
```python

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
new_train_gen = trainGenerator(2,'dataset_capseq/train', 'imgs', 'masks',data_gen_args,)
new_test_gen = trainGenerator(1,'dataset_capseq/test', 'imgs', 'masks', {},)
```
3) Примерный код обучения модели 
``` python
kaggle_model = kaggle_unet()
kaggle_history = kaggle_model.fit_generator(new_train_gen,
                              steps_per_epoch=2000,
                              epochs=5,
                              callbacks=[kaggle_model_checkpoint],
                              validation_data = new_test_gen,
                              validation_steps = 10,
                                           )
```
