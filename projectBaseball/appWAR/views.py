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
    pit_path = os.path.dirname('C:/Users/sync7/test/Djangovenv/projectBaseball/appWAR/model/') + '\pitcher_WAR_allData.h5'
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


        #92의 WAR값으로 변환
        #-2.xx~11.xx까지 분포 -2 에서 8까지 있다고 봐도 무방 (최소: -2.27, 최대 11.73)
        #+12를 100으로 환산한다
        if y_pred < 0 :
            transformed_val = y_pred * 0
        elif y_pred >= 11.73:
            transformed_val = 100
        else:
            transformed_val = y_pred * 8.33
            transformed_val = transformed_val.astype(int)

        ls=list()
        for i in range(len(str(list(y_pred)[0]))):
            ls.append(str(list(y_pred)[0])[i])

        # 다시 html파일로 넘겨준다
        hitter_predict = {'data': list(y_pred), 'transformed_data':list(transformed_val), 'ls':ls}

        return render(request, 'appWAR/result_hitter.html', hitter_predict)

    #Select에서 Pitcher를 선택했을 경우
    elif request.POST.get('hitterPitcher') == "selectPitcher":

        train_path = os.path.dirname('C:/Users/sync7/test/Djangovenv/projectBaseball/appWAR/model/') + '\Pitcher_X_train_allData.csv'
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


        # 92의 WAR값으로 변환
        # -1.xx~8.xx까지 분포 -1 에서 8까지 있다고 봐도 무방 (max:8.49, min:-1.35)
        if y_pred<0:
            transformed_val = 0
        elif y_pred >= 8.49:
            transformed_val = 100
        else:
            transformed_val = y_pred * 11.1
            transformed_val = transformed_val.astype(int)

        ls = list()
        for i in range(len(str(list(y_pred)[0]))):
            ls.append(str(list(y_pred)[0])[i])

        #다시 html로 값을 넘겨준다
        pitcher_predict = {'data': list(y_pred), 'transformed_data': list(transformed_val), 'ls':ls}


        return render(request, 'appWAR/result_pitcher.html', pitcher_predict)

    return render(request, 'appWAR/WAR.html')



