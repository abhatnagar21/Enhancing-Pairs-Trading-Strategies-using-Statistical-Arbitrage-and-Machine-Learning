![image](https://github.com/user-attachments/assets/d7d0a5e2-d786-4c00-8160-b2a83e8b2d79)
![image](https://github.com/user-attachments/assets/a5fd6f69-ffa1-4b49-90b9-1fd77ac4a472)
![image](https://github.com/user-attachments/assets/96de4ddf-c8dd-437e-af5e-9067a3cfdfe0)
![image](https://github.com/user-attachments/assets/881baf72-b076-49bc-baa9-de61525fc4e1)
![image](https://github.com/user-attachments/assets/0d8376d4-3227-488b-8236-b47dc96c80e1)
Linear Regression Predictions vs Actual:
            Actual Spread  Predicted Spread
Date                                       
2023-05-26     154.476028        148.866540
2023-05-30     150.964783        153.190486
2023-05-31     148.232452        149.783992
2023-06-01     149.550354        147.133175
2023-06-02     151.479630        148.411760
2023-06-05     153.370575        150.283479
2023-06-06     151.507858        152.118012
2023-06-07     142.725021        150.310866
2023-06-08     141.853302        141.790047
2023-06-09     142.975998        140.944334

Linear Regression - MSE: 17.4542, MAE: 3.2313, Percentage Error: 2.07%

Random Forest Regressor Predictions vs Actual:
            Actual Spread  Predicted Spread
Date                                       
2023-05-26     154.476028        141.709361
2023-05-30     150.964783        141.709361
2023-05-31     148.232452        141.709361
2023-06-01     149.550354        141.709361
2023-06-02     151.479630        141.709361
2023-06-05     153.370575        141.709361
2023-06-06     151.507858        141.709361
2023-06-07     142.725021        141.709361
2023-06-08     141.853302        136.771473
2023-06-09     142.975998        139.666185

Random Forest Regressor - MSE: 426.7790, MAE: 15.3640, Percentage Error: 9.18%

Support Vector Regression Predictions vs Actual:
            Actual Spread  Predicted Spread
Date                                       
2023-05-26     154.476028        120.909870
2023-05-30     150.964783        116.454159
2023-05-31     148.232452        119.876234
2023-06-01     149.550354        122.938168
2023-06-02     151.479630        121.434420
2023-06-05     153.370575        119.330109
2023-06-06     151.507858        117.449100
2023-06-07     142.725021        119.300545
2023-06-08     141.853302        128.930935
2023-06-09     142.975998        129.706386

Support Vector Regression - MSE: 1725.8544, MAE: 34.5138, Percentage Error: 21.05%
/usr/local/lib/python3.11/dist-packages/keras/src/layers/rnn/rnn.py:200: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
WARNING:tensorflow:5 out of the last 11 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x7916bf3b6840> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
1/5 ━━━━━━━━━━━━━━━━━━━━ 0s 241ms/stepWARNING:tensorflow:5 out of the last 11 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x7916bf3b6840> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
5/5 ━━━━━━━━━━━━━━━━━━━━ 1s 102ms/step

LSTM Predictions vs Actual:
            Actual Spread  Predicted Spread
Date                                       
2023-05-26     154.476028        150.993652
2023-05-30     150.964783        155.514130
2023-05-31     148.232452        151.953278
2023-06-01     149.550354        149.179794
2023-06-02     151.479630        150.517853
2023-06-05     153.370575        152.475632
2023-06-06     151.507858        154.393463
2023-06-07     142.725021        152.504288
2023-06-08     141.853302        143.581650
2023-06-09     142.975998        142.694550

LSTM - MSE: 16.1456, MAE: 2.9817, Percentage Error: 1.94%

Comparative Analysis of ML Models:
                                   MSE        MAE  Percentage Error (%)
Linear Regression            17.454223   3.231350              2.073885
Random Forest Regressor     426.779036  15.364021              9.179251
Support Vector Regression  1725.854377  34.513793             21.049766
LSTM                         16.145647   2.981676              1.943691
