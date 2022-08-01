import os

from django.shortcuts import render

# Create your views here.
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def WAR(request):

    #Model Load (Hitter, Pitcher)
    hit_path = os.path.dirname('C:/Users/sync7/test/Djangovenv/projectBaseball/appWAR/model/') + '\hitter_WAR.h5'
    hitterModel = tf.keras.models.load_model(hit_path)
    pit_path = os.path.dirname('C:/Users/sync7/test/Djangovenv/projectBaseball/appWAR/model/') + '\pitcher_WAR.h5'
    pitcherModel = tf.keras.models.load_model(pit_path)

    #Select에서 Hitter를 선택했을 경우
    if request.POST.get('hitterPitcher') == "selectHitter":

        #min-max scaler 적용을 위해 data 불러와서 적용
        train_path = os.path.dirname('C:/Users/sync7/test/Djangovenv/projectBaseball/appWAR/model/') + '\X_train.csv'
        hitter_train = pd.read_csv(train_path, index_col=0)
        minmax_scaler = MinMaxScaler()
        hitter_train_scaled = minmax_scaler.fit_transform(np.array(hitter_train))

        #WAR.html로부터 one~eleven 값 받아옴
        one = request.POST.get('one')
        two = request.POST.get('two')
        three = request.POST.get('three')
        four = request.POST.get('four')
        five = request.POST.get('five')
        six = request.POST.get('six')
        seven = request.POST.get('seven')
        eight = request.POST.get('eight')
        nine = request.POST.get('nine')
        ten = request.POST.get('ten')
        eleven = request.POST.get('eleven')

        #받아온 값으로부터 X_test array 생성
        hitter_test = np.array([[float(one),float(two),float(three),float(four),float(five),
                            float(six),float(seven),float(eight),float(nine),float(ten),float(eleven)]])

        #X_test값 minmax_scaler에 적용
        hitter_test_scaled = minmax_scaler.transform(hitter_test)

        #예측값 받아옴 (minmax-scaler 적용한 것을 되돌려준다, 소수점 2번째 자리에서 반올림)
        y_pred = np.round(list(np.exp(hitterModel.predict(hitter_test_scaled).ravel()) - 1), 2)

        #다시 html파일로 넘겨준다
        hitter_predict = {'data' : list(y_pred)}

        return render(request, 'appWAR/WAR_practice.html', hitter_predict)

    #Select에서 Pitcher를 선택했을 경우
    elif request.POST.get('hitterPitcher') == "selectPitcher":

        train_path = os.path.dirname('C:/Users/sync7/test/Djangovenv/projectBaseball/appWAR/model/') + '\Pitcher_X_train.csv'
        pitcher_train = pd.read_csv(train_path, index_col=0)
        minmax_scaler = MinMaxScaler()
        pitcher_train_scaled = minmax_scaler.fit_transform(np.array(pitcher_train))

        one2 = request.POST.get('one')
        two2 = request.POST.get('two')
        three2 = request.POST.get('three')
        four2 = request.POST.get('four')
        five2 = request.POST.get('five')
        six2 = request.POST.get('six')
        seven2 = request.POST.get('seven')
        eight2 = request.POST.get('eight')
        nine2 = request.POST.get('nine')
        ten2 = request.POST.get('ten')
        eleven2 = request.POST.get('eleven')
        twelve2 = request.POST.get('twelve')

        pitcher_test = np.array([[float(one2), float(two2), float(three2), float(four2), float(five2),
                            float(six2), float(seven2), float(eight2), float(nine2), float(ten2),
                            float(eleven2), float(twelve2)]])

        # X_test값 minmax_scaler에 적용
        pitcher_test_scaled = minmax_scaler.transform(pitcher_test)

        # 예측값 받아옴 (minmax-scaler 적용한 것을 되돌려준다, 소수점 2번째 자리에서 반올림)
        y_pred = np.round(list(np.exp(pitcherModel.predict(pitcher_test_scaled).ravel()) - 1), 2)

        pitcher_predict = {'data': list(y_pred)}

        return render(request, 'appWAR/WAR_practice.html', pitcher_predict)

    return render(request, 'appWAR/WAR.html')

def WAR_practice(request):
    return render(request, 'appWAR/WAR_practice.html')

