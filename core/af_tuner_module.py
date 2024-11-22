import copy
import json
import math
import pandas as pd
from geopy.distance import distance
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager
from catboost import CatBoostRegressor
import joblib
from typing import List


class AntennaFeederTuner:
    """
    load_model() -- Загрузка модели
    create_dataset() -- Создание DataSet для обучения модели прогноза RSRP
    predict_nearest_rsrp() -- Прогноз RSRP от ближайшей антенны
    call_sinr() -- Расчёт SINR для точки по ближайшей антенны
    call_sinr_v2() -- Расчёт SINR для точки по формуле мощности сигнала антенны
    call_sinr_v3() -- Расчёт SINR для точки по формуле мощности сигнала антенны с учётом RET
    call_rsrp_neg(center_lng, center_lat): -- Расчёт знаменателя для вычисления SINR

    get_max_power_id() -- Нахождение антенны с максимальным сигналом / или всех отсортированных по

    """

    def __init__(self, band: str = None):
        # Справочник антенн с K усиления в зависимости от угла
        with open('./sourse_data/antennas_gains.dat', 'rb') as f:
            self.antennas_gains = json.loads(f.read())
        # Справочник антенн с K усиления в зависимости от угла наклона
        with open('./sourse_data/antennas_gains_vertical.dat', 'rb') as f:
            self.antennas_gains_tilt = json.loads(f.read())

        # Справочник антенн с их параметрами
        # self.afu_file_name = './sourse_data/Тихвин_соты_30.09.2024.xlsx'
        # self.afu_file_name = './sourse_data/Тихвин_соты_30.09.2024_1800_6ant.xlsx'  # 6 антенн
        self.afu_file_name = './sourse_data/Тихвин_соты_30.09.2024_1800_RET_V2.xlsx'  # RET

        self.dc_anten = self._get_dict_antens(band)
        self.dc_anten_v2 = self._get_dict_antens_v2(band)
        self.thermal_noise = -122  # Значени теплового шума
        # Список квадратов с их координатами центов и RSRP_max в этом квадрате
        self.df_max_rsrp_from_squar: pd.DataFrame = pd.read_csv('./sourse_data/to_dataset_5x5_1800.csv')
        # print(f'{self.df_max_rsrp_from_squar.shape=}')

        # Расчет корректирующего коэффициента a(h_m) для высоты антенны мобильной станции дБ
        self.a_hm = self.calculate_a_hm()

        # Проверка наличия антенн в справочнике
        # for key, ls_antenn in self.dc_anten.items():
        #     for dc_antenn in ls_antenn:
        #         for key_ant, val_ant in dc_antenn.items():
        #             if key_ant == 'ANTENNA_NAME':
        #                 print(val_ant)
        #                 signal_atten = self.antennas_gains[str(val_ant)]

    def load_model(self, model_folder: str, model_name: str):
        """ Функция загрузки модели"""
        print('Загрузка модели.')
        self.model = CatBoostRegressor()
        self.model.load_model(f'./{model_folder}/{model_name}')
        self.scaler = joblib.load(f'./{model_folder}/scaler.pkl')

    def call_rsrp_neg(self, center_lng, center_lat, antenn_id):
        """ Расчёт интерференции помех в ваттах

        1. Находим соседние антенны.
        2. Нахождение фичей соседних антенн
        3. Для каждой соседней антенны делаем прогноз RSRP
        4. Возвращаем rsrp соседей для точки

        :param center_lng: Широта точки
        :param center_lat: Долгота точки

        """
        # Нахождение соседних антенн по id домашней
        neighbors = self.get_neighbors_antenna(center_lat=center_lat, center_lng=center_lng, dc_anten=self.dc_anten,
                                               antenn_id=antenn_id)
        interference_w = 0.0
        # Прогноз SINR
        for antenn_id in neighbors:
            # прогноз RSRP
            y_pred = self.predict_nearest_rsrp_v2(point_lng=center_lng, point_lat=center_lat, antenn_id=antenn_id)
            inter_0 = 10 ** ((y_pred - 30) / 10)  # RSRP дБм -> ватты
            interference_w += inter_0  # суммирование помех в ваттах
        return interference_w

    def call_sinr(self, rsrp_max, center_lng, center_lat):
        """ Расчёт SINR для точки по ближайшей антенны
        1. Находим соседние антенны.
        2. Нахождение фичей соседних антенн
        3. Для каждой соседней антенны делаем прогноз RSRP
        4. Вычисляем SINR

        :param rsrp_max: Максимальное значение RSRP в точке
        :param center_lng: Широта точки
        :param center_lat: Долгота точки
        """
        # Нахождение соседних антенн
        neighbors = self.get_neighbors_antenna(center_lat=center_lat, center_lng=center_lng, dc_anten=self.dc_anten)
        interference_w = 0.0
        # Прогноз SINR
        for key in neighbors:
            dc_val = neighbors[key]
            ds = dc_val['ds']
            perp = dc_val['perp']
            angle = dc_val['angle']
            antenna_name = dc_val['antenna_name']

            # Рассчитываем эффективное расстояние
            eff_distance = self.effect_distance(angle_radians=angle, shortest_distance=ds)
            # Угол в градусах
            angle_gr = int(np.rad2deg(angle))
            # Ослабление сигнала на данном угле
            signal_atten = self.antennas_gains[str(antenna_name)][str(angle_gr)]

            dc_result = {
                'ds': ds,
                'perp': int(perp),
                'angle': angle_gr,
                'eff_distance': int(eff_distance),
                'signal_atten': signal_atten
            }
            # Dict -> DataFrame
            df = pd.DataFrame([dc_result])
            # Нормализация данных
            ls_columns = ['ds', 'perp', 'angle', 'eff_distance', 'signal_atten']
            df[ls_columns] = self.scaler.transform(df[ls_columns])
            row_to_predict = df.iloc[0]
            # Прогноз
            y_pred = int(self.model.predict(row_to_predict))  # прогноз RSRP
            inter_0 = 10 ** ((y_pred - 30) / 10)  # RSRP дБм -> ватты
            interference_w += inter_0  # суммирование помех в ваттах

        signal_level = 10 ** ((rsrp_max - 30) / 10)  # мощность сигнала в ваттах
        th_noise_w = 10 ** ((self.thermal_noise - 30) / 10)  # тепловой шум в ваттах

        sinr_p = signal_level / (interference_w + th_noise_w)  # SINR в ваттах
        sinr = int(10 * np.log10(sinr_p))  # ватты -> dB
        # !!! Для отладки
        # interference_db = int(10 * np.log10(interference_w) + 30)  # величина помех в dBm
        # print('-' * 30)
        # print(f'RSRP_w: {signal_level}')
        # print(f'interference_w: {interference_w}')
        # print(f'th_noise_w: {th_noise_w}')
        # print('')
        # print(f'RSRP_db: -105')
        # print(f'interference_db: {interference_db}')
        # print(f'th_noise_db: -122')
        # print('')
        # print(f'SINR: {sinr_p}W, {sinr}dB')
        return sinr

    def call_sinr_v2(self, center_lng, center_lat, ls_id_antenn, uses_tilt: bool = False):
        """ Расчёт SINR для точки по формуле мощности сигнала антенны

        :param center_lng: Координаты точки расчёта
        :param center_lat: Координаты точки расчёта
        :param ls_id_antenn: Список id антенн
        :param uses_tilt: Использовать данные угла наклона антенны
        :return: SINR в dB
        """
        id_antenn_max = ls_id_antenn[0]
        chisl_wat = 0
        znam_wat = 0

        for id_antenn in ls_id_antenn:
            power_db = self.calculate_power_id(center_lng=center_lng,
                                               center_lat=center_lat,
                                               antenn_id=id_antenn,
                                               new_angle=400,
                                               uses_tilt = uses_tilt)
            if id_antenn == id_antenn_max:
                chisl_wat += self.dbm_2_wat(power_db)
                chisl_db = power_db
            else:
                znam_wat += self.dbm_2_wat(power_db)

        sinr_wat = chisl_wat / (znam_wat + self.dbm_2_wat(-122))
        sinr_db = self.wat_2_db(sinr_wat)
        # return (id_antenn_max, chisl_db, sinr_db)
        return sinr_db

    def predict_nearest_rsrp_v2(self, point_lng, point_lat, antenn_id) -> int:
        """ Прогноз RSRP в точке с заданными координатами от антенны c данным ID

        :param point_lng:
        :param point_lat:
        :param antenn_id:
        :return:
        """

        nearest_antenna = self.get_nearest_antenna_v3(center_lng=point_lng, center_lat=point_lat, antenn_id=antenn_id)
        antenna_name = nearest_antenna['antenna_name']
        ds = nearest_antenna['ds']
        # Угол между векторами в радианах и градусах
        agle = nearest_antenna['angle_between']
        agle_grad = int(np.rad2deg(agle))
        # Рассчитываем эффективное расстояние
        eff_distance = self.effect_distance(angle_radians=agle, shortest_distance=ds)
        # Ослабление сигнала на данном углу
        signal_atten = self.antennas_gains[str(antenna_name)][str(agle_grad)]

        dc_result = {
            'ds': ds,
            'perp': int(nearest_antenna['perp']),
            'angle': agle_grad,
            'eff_distance': int(eff_distance),
            'signal_atten': signal_atten
        }
        # if debug:
        #     print(json.dumps(dc_result))

        # Dict -> DataFrame
        df = pd.DataFrame([dc_result])
        # Нормализация данных
        ls_columns = ['ds', 'perp', 'angle', 'eff_distance', 'signal_atten']
        df[ls_columns] = self.scaler.transform(df[ls_columns])
        row_to_predict = df.iloc[0]
        # Прогноз
        y_pred = int(self.model.predict(row_to_predict))  # прогноз RSRP
        return y_pred

    def predict_nearest_rsrp(self, point_lng, point_lat, debug: bool = False, rotate: int = 0, ant_rotate_id: int = 0):
        """ Прогноз RSRP от ближайшей по расстоянию антенны

        :param point_lng: Широта точки
        :param point_lat: Долгота точки
        :param debug: Флаг отладки
        :param rotate: Поворот азимута в градусах
        :param ant_rotate_id: ID антенны, которую поворачиваем
        :return: y_pred
        """
        # Получение данных ближайшей антенны
        nearest_antenna = self.get_nearest_antenna_v2(center_lng=point_lng,
                                                      center_lat=point_lat, dc_anten=self.dc_anten)
        antenna_name = nearest_antenna['antenna_name']
        antenna_id = nearest_antenna['antenna_id']
        if rotate and antenna_id == ant_rotate_id:
            old_azimuth = nearest_antenna['azimuth']
            # Поворот против часовой стрелки
            new_azimuth = old_azimuth - rotate
            # print(f'{old_azimuth=}, {new_azimuth=}')
        else:
            rotate = 0
        # Поворот антенны против часовой стрелки
        nearest_antenna.update({'azimuth': nearest_antenna['azimuth'] - rotate})
        # try:
        dc_distance_new = self.call_distance(**nearest_antenna)
        perp = dc_distance_new['perp']
        ds = dc_distance_new['ds']
        # except Exception as e:
        #     print(f'{dc_distance_new=}')
        #     print(f'{nearest_antenna=} Error: {e}')

        # Угол между векторами в радианах и градусах
        agle = dc_distance_new['angle_between']
        agle_grad = int(np.rad2deg(agle))

        # Рассчитываем эффективное расстояние
        eff_distance = self.effect_distance(angle_radians=agle, shortest_distance=ds)
        # Ослабление сигнала на данном углу
        signal_atten = self.antennas_gains[str(antenna_name)][str(agle_grad)]

        dc_result = {
            'ds': ds,
            'perp': int(perp),
            'angle': agle_grad,
            'eff_distance': int(eff_distance),
            'signal_atten': signal_atten
        }
        if debug:
            print(json.dumps(dc_result))

        # Dict -> DataFrame
        df = pd.DataFrame([dc_result])
        # Нормализация данных
        ls_columns = ['ds', 'perp', 'angle', 'eff_distance', 'signal_atten']
        df[ls_columns] = self.scaler.transform(df[ls_columns])
        row_to_predict = df.iloc[0]
        # Прогноз
        y_pred = int(self.model.predict(row_to_predict))  # прогноз RSRP
        return y_pred

    def create_dataset(self, out_file_name):
        """ Создание DataSet для обучения модели прогноза RSRP

        """
        # Создаем пустой DataFrame с необходимыми колонками
        columns = ['id_squar', 'rsrp', 'ds', 'perp', 'angle', 'eff_distance',
                   'signal_atten', 'antenna_id', 'antenna_name']
        pd.DataFrame(columns=columns).to_csv(out_file_name, index=False, mode='w')

        # Подготовим задачи для передачи в пул процессов
        tasks = [(row.id_squar, row.latitude, row.longitude, row.servingcellrsrp)
                 for row in self.df_max_rsrp_from_squar.itertuples()]

        # Для параллельной обработки используем multiprocessing
        with Manager() as manager:
            lock = manager.Lock()
            pool = Pool(initializer=self.init_pool, initargs=(lock,))

            # Параллельная обработка строк с отображением прогресса
            list(tqdm(pool.imap_unordered(self.create_dataset_one_row_wrapper, tasks), total=len(tasks)))
            pool.close()
            pool.join()

    @staticmethod
    def init_pool(shared_lock):
        global lock
        lock = shared_lock

    def create_dataset_one_row_wrapper(self, args):
        """ Метод обёртка.
        Когда вы создаете пул процессов с multiprocessing.Pool, каждый процесс запускается независимо от других.
        Чтобы процессы могли обмениваться данными, Python использует механизм сериализации (через модуль pickle).
        Однако pickle не может сериализовать (или "упаковать") метод экземпляра класса,
        такой как self.create_dataset_one_row, потому что он связан с объектом класса.

        Чтобы обойти эту проблему, используется обёртка. Обёртка превращает метод экземпляра класса в обычную
        функцию, которую Python может передавать между процессами. Вот что делает обёртка:
        """
        return self.create_dataset_one_row(*args)

    def create_dataset_one_row(self, id_squar, latitude, longitude, servingcellrsrp):
        # Преобразование аргументов в формат для обработки
        id_squar = int(id_squar)
        center_lat = float(latitude)
        center_lng = float(longitude)

        # Выбираем ближайшую антенну
        nearest_antenna = self.get_nearest_antenna(center_lng=center_lng, center_lat=center_lat, dc_anten=self.dc_anten)
        nearest_antenna_name = nearest_antenna['antenna_name']
        nearest_antenna_id = nearest_antenna['antenna_id']

        # Рассчитываем эффективное расстояние
        eff_distance = self.effect_distance(angle_radians=nearest_antenna['angle'],
                                            shortest_distance=nearest_antenna['ds'])

        # Угол в градусах
        angle_gr = int(np.rad2deg(nearest_antenna['angle']))

        # Ослабление сигнала на данном угле
        signal_atten = self.antennas_gains[str(nearest_antenna_name)][str(angle_gr)]

        # Формируем строку для записи
        dc_result = {
            'id_squar': id_squar,
            'rsrp': int(servingcellrsrp),
            # 'rsrq': int(servingcellrsrq),
            'ds': nearest_antenna['ds'],
            'perp': int(nearest_antenna['perp']),
            'angle': angle_gr,
            'eff_distance': int(eff_distance),
            'signal_atten': signal_atten,
            'antenna_id': nearest_antenna_id,  # Ближайшей антенны
            'antenna_name': nearest_antenna_name,  # Ближайшей антенны
            # 'P_transmit': P_transmit
        }

        # Используем блокировку для синхронизированной записи в файл
        with lock:
            df_row = pd.DataFrame([dc_result])
            df_row.to_csv('output_rsrp_5m_max_1800.csv', mode='a', header=False, index=False)

    def get_neighbors_antenna(self, center_lng, center_lat, dc_anten, antenn_id):
        """ Получение данных соседних антенн

        :param antenn_id:
        :param center_lng:
        :param center_lat:
        :param dc_anten:
        :return:
        """
        # Получение данных всех антенны
        nearest_antenna = self.get_nearest_antenna(center_lng, center_lat, dc_anten, all=True)
        del nearest_antenna[antenn_id]
        return nearest_antenna

    def get_nearest_antenna_v3(self, center_lng, center_lat, antenna_id):
        """ Получение данных антенны по её ID

        :param center_lng:
        :param center_lat:
        :param dc_anten:
        :param antenn_id:
        :return: {'antenna_id': 14, 'antenna_name': 'G3WD-21', 'center_lng': 33.47165,
         'center_lat': 59.63186, 'anten_lng': 33.472333, 'anten_lat': 59.631325, 'azimuth': 20,
          'angle_between': 1.2553819009681213, 'ds': 70, 'perp': 66}

        """
        # В dc_antenn_fn собираем антенны с их: ds, perp, angle_between для нахождения ближайшей
        dc_antenn_fn = self.dc_anten_v2[antenna_id]

        dc_antenn_fn.update({'center_lng': center_lng, 'center_lat': center_lat})
        # Находим: ds, angle_between, perp
        dc_distance = self.call_distance(**dc_antenn_fn)
        angle_between = dc_distance['angle_between']
        angle_bet_grad = dc_distance['angle_bet_grad']
        ds = dc_distance['ds']
        perp = dc_distance['perp']
        dc_antenn_fn.update({'angle_between': angle_between,
                             'angle_bet_grad': int(angle_bet_grad),
                             'ds': ds,
                             'perp': perp})
        # print(dc_antenn_fn[antenn_id])
        return dc_antenn_fn

    def get_nearest_antenna_v2(self, center_lng, center_lat, dc_anten):
        """ Получение данных ближайшей по расстоянию антенны

        :param center_lng:
        :param center_lat:
        :param dc_anten:
        :return: {'antenna_name': 'G4WD-21', 'AZIMUTH': 0.32373758881581166}
        """
        cn_91_gr = 0
        # В dc_antenn_fn собираем антенны с их: ds, perp, angle_between для нахождения ближайшей
        dc_antenn_fn = {}

        for tower_name, ls_antenn in dc_anten.items():
            anten_lng = ls_antenn[0]['coord_long']
            anten_lat = ls_antenn[0]['coord_lat']

            for dc_antenn in ls_antenn:
                antenna_id = dc_antenn['antenna_id']
                antenna_name = dc_antenn['ANTENNA_NAME']
                azimuth = dc_antenn['AZIMUTH']
                kwargs = {'azimuth': azimuth,
                          'center_lng': center_lng,
                          'center_lat': center_lat,
                          'anten_lng': anten_lng,
                          'anten_lat': anten_lat}

                dc_distance = self.call_distance(**kwargs)
                angle_between = dc_distance['angle_between']
                if np.rad2deg(angle_between) > 90:
                    cn_91_gr += 1
                    continue

                ds = dc_distance['ds']
                perp = dc_distance['perp']
                dc_antenn_fn[antenna_id] = {'antenna_id': antenna_id,
                                            'antenna_name': antenna_name,
                                            'center_lng': center_lng,
                                            'center_lat': center_lat,
                                            'anten_lng': anten_lng,
                                            'anten_lat': anten_lat,
                                            'azimuth': azimuth,
                                            'ds': ds,
                                            'perp': perp,
                                            'angle': angle_between
                                            }

        # Фильтруем антенны, у которых 'perp' не равно None
        filtered_antennas = {k: v for k, v in dc_antenn_fn.items() if v['perp'] is not None}
        # Если таких антенн нет, возвращаем None
        if not filtered_antennas:
            return None

        # Находим антенну с минимальным 'ds'
        min_ds = min(filtered_antennas.values(), key=lambda x: x['ds'])['ds']
        candidates = {k: v for k, v in filtered_antennas.items() if v['ds'] == min_ds}

        # Находим антенну с минимальным 'angle' среди кандидатов
        min_angle = min(candidates.values(), key=lambda x: x['angle'])['angle']
        final_candidates = {k: v for k, v in candidates.items() if v['angle'] == min_angle}

        # Возвращаем первую антенну из списка финальных кандидатов
        id_ant = next(iter(final_candidates))
        return dc_antenn_fn[id_ant]

    def get_antenns_params(self, center_lng, center_lat):
        """ Получение данных всех антенн + ds, angle
        !!! Антенны стоящие "спиной" тоже учитываются.
        Используется для нахождение антенны с максимальным сигналом

        :param center_lng: Координаты точки в квадрате
        :param center_lat: Координаты точки в квадрате
        :param dc_anten:
        :param all: Флаг -- выгрузка всех антен
        :return: {id: {'BAND': 'LTE1800', 'ANTENNA_NAME': 'APX16DWV-16DWVL-C', 'azimuth': 85, 'tilt': 3, 'HBW': 63,
                     'anten_lng': 33.5075, 'anten_lat': 59.638421, 'height': 38, 'power': 41.2,
                     'center_lng': 33.4699656677246, 'center_lat': 59.6640264892578, 'angle_between': 2.45565455324097,
                     'angle_bet_grad': 140, 'ds': 3551, 'perp': None}, ..}
        """
        dc_anten_v2 = self.dc_anten_v2
        # В dc_antenn_fn собираем антенны с их: ds, perp, angle_between для нахождения ближайшей
        dc_antenn_fn = copy.copy(dc_anten_v2)

        for antenna_id, dc_antenn in dc_anten_v2.items():
            dc_antenn_fn[antenna_id].update({'center_lng': center_lng, 'center_lat': center_lat})
            # Находим: ds, angle_between, perp
            dc_distance = self.call_distance(**dc_antenn_fn[antenna_id])
            angle_between = dc_distance['angle_between']
            angle_bet_grad = dc_distance['angle_bet_grad']
            ds = dc_distance['ds']
            perp = dc_distance['perp']
            dc_antenn_fn[antenna_id].update({'angle_between': angle_between,
                                             'angle_bet_grad': int(angle_bet_grad),
                                             'ds': ds,
                                             'perp': perp})
        return dc_antenn_fn

    def get_nearest_antenna(self, center_lng, center_lat, all: bool = False):
        """ Получение данных всех / или ближайшей по расстоянию антенны + ds, perp, angle

        :param center_lng: Координаты точки в квадрате
        :param center_lat: Координаты точки в квадрате
        :param dc_anten:
        :param all: Флаг -- выгрузка всех антен
        :return: {'antenna_name': 'G4WD-21', 'ds': 2427, 'perp': 772, 'angle': 0.32373758881581166}
        """
        cn_91_gr = 0
        # В dc_antenn_fn собираем антенны с их: ds, perp, angle_between для нахождения ближайшей
        dc_antenn_fn = {}

        for tower_name, ls_antenn in self.dc_anten.items():
            anten_lng = ls_antenn[0]['coord_long']
            anten_lat = ls_antenn[0]['coord_lat']

            for dc_antenn in ls_antenn:
                antenna_id = dc_antenn['antenna_id']
                antenna_name = dc_antenn['ANTENNA_NAME']
                azimuth = dc_antenn['AZIMUTH']
                height = dc_antenn['HEIGHT']
                power = dc_antenn['POWER']
                kwargs = {'azimuth': azimuth,
                          'center_lng': center_lng,
                          'center_lat': center_lat,
                          'anten_lng': anten_lng,
                          'anten_lat': anten_lat}

                dc_distance = self.call_distance(**kwargs)
                angle_between = dc_distance['angle_between']
                if np.rad2deg(angle_between) > 90:
                    cn_91_gr += 1
                    continue

                ds = dc_distance['ds']
                perp = dc_distance['perp']

                dc_antenn_fn[antenna_id] = {'antenna_id': antenna_id,
                                            'antenna_name': antenna_name,
                                            'ds': ds,
                                            'perp': perp,
                                            'angle': angle_between,
                                            'height': height,
                                            'power': power}

        # Фильтруем антенны, у которых 'perp' не равно None
        filtered_antennas = {k: v for k, v in dc_antenn_fn.items() if v['perp'] is not None}
        # Если таких антенн нет, возвращаем None
        if not filtered_antennas:
            return None

        # Отладка
        # if len(list(filtered_antennas)) == 33:
        #     print(f'{len(list(filtered_antennas))}')

        if all:
            return filtered_antennas

        # Находим антенну с минимальным 'ds'
        min_ds = min(filtered_antennas.values(), key=lambda x: x['ds'])['ds']
        candidates = {k: v for k, v in filtered_antennas.items() if v['ds'] == min_ds}

        # Находим антенну с минимальным 'angle' среди кандидатов
        min_angle = min(candidates.values(), key=lambda x: x['angle'])['angle']
        final_candidates = {k: v for k, v in candidates.items() if v['angle'] == min_angle}

        # Возвращаем первую антенну из списка финальных кандидатов
        id_ant = next(iter(final_candidates))
        return dc_antenn_fn[id_ant]

    def effect_distance(self, angle_radians, shortest_distance):
        """
        Функция для расчёта эффективного расстояния с учётом угла.
        Если угол > 90 градусов (угол > pi/2 радиан), то ячейка находится за антенной,
        и эффективное расстояние будет увеличено.

        :param angle_radians: Угол между направлением антенны и направлением на ячейку в радианах.
        :param shortest_distance: Кратчайшее расстояние до антенны (ps_x)
        :return: Эффективное расстояние
        """
        # Если угол больше 90 градусов (pi/2 радиан), то ячейка за антенной
        if angle_radians >= math.pi / 2:
            # Увеличиваем эффективное расстояние, используя поправочный коэффициент
            # Можно применить масштабирование для углов за антенной
            correction_factor = 10  # Коэффициент для ослабления сигнала (подбирается)
            effective_distance = shortest_distance * correction_factor
        else:
            # Стандартный расчёт для углов < 90 градусов
            effective_distance = shortest_distance / math.cos(angle_radians)
        return effective_distance

    def _get_dict_antens_v2(self, band: str = None):
        """ Создаёт справочник антенн с их координатами

        :params band: Частотный диапазон антенны
        :return:{'Tower_1': [{'antenna_id': 27, 'ANTENNA_NAME': 'G3WD-21', 'coord_long': 33.472333, 'coord_lat': 59.631325,
                    'AZIMUTH': 235, 'tilt': 3, 'HBW': 67, 'BAND': 'LTE1800'}, ..], ..}
        """
        df_anten = pd.read_excel(self.afu_file_name)
        df_anten = df_anten[~df_anten['CELL_LON_COORD'].isnull()]
        df_anten: pd.DataFrame = df_anten[
            ['ANTENNA_ID', 'BAND', 'ANTENNA_NAME', 'AZIMUTH', 'HBW', 'CELL_LON_COORD', 'CELL_LAT_COORD', 'HEIGHT',
             'POWER', 'TILT', 'RET']]
        # Фильтр антенн по их частотному диапазону
        if band:
            df_anten = df_anten[df_anten['BAND'] == band]
        dc_anten = dict()
        for row in df_anten.itertuples():
            power = float(row.POWER.replace(',', '.')) if isinstance(row.POWER, str) else row.POWER
            height = float(row.HEIGHT.replace(',', '.')) if isinstance(row.HEIGHT, str) else row.HEIGHT

            anten_lng = float(row.CELL_LON_COORD.replace(',', '.')) if isinstance(row.CELL_LON_COORD,
                                                                                  str) else row.CELL_LON_COORD
            anten_lat = float(row.CELL_LAT_COORD.replace(',', '.')) if isinstance(row.CELL_LAT_COORD,
                                                                                  str) else row.CELL_LAT_COORD

            dc_anten[row.ANTENNA_ID] = {'BAND': row.BAND,
                                        'ANTENNA_NAME': row.ANTENNA_NAME,
                                        'azimuth': row.AZIMUTH,
                                        'tilt': row.TILT,
                                        'HBW': row.HBW,
                                        'anten_lng': anten_lng,
                                        'anten_lat': anten_lat,
                                        'height': height,
                                        'power': power,
                                        'ret': row.RET}
        return dc_anten

    def _get_dict_antens(self, band: str = None):
        """ Создаёт справочник антенн с их координатами

        :params band: Частотный диапазон антенны
        :return:{'Tower_1': [{'antenna_id': 27, 'ANTENNA_NAME': 'G3WD-21', 'coord_long': 33.472333, 'coord_lat': 59.631325,
                    'AZIMUTH': 235, 'HBW': 67, 'BAND': 'LTE1800'}, ..], ..}
        """
        df_anten = pd.read_excel(self.afu_file_name)
        df_anten = df_anten[~df_anten['CELL_LON_COORD'].isnull()]
        df_anten = df_anten[
            ['ANTENNA_ID', 'BAND', 'ANTENNA_NAME', 'AZIMUTH', 'HBW', 'CELL_LON_COORD', 'CELL_LAT_COORD', 'HEIGHT',
             'POWER']]
        # Фильтр антенн по их частотному диапазону
        if band:
            df_anten = df_anten[df_anten['BAND'] == band]
        # Группировка координат антенн, для идентификации вышек
        gp_anten = df_anten[['CELL_LON_COORD', 'CELL_LAT_COORD']]
        gp_anten = gp_anten.groupby(['CELL_LON_COORD', 'CELL_LAT_COORD']).mean().reset_index()

        dc_anten = dict()
        # Перебор сгруппированных координат
        for i, (index, row) in enumerate(gp_anten.iterrows(), start=1):
            str_long = row['CELL_LON_COORD']
            str_lat = row['CELL_LAT_COORD']
            float_long = float(str_long.replace(',', '.')) if isinstance(str_long, str) else str_long
            float_lat = float(str_lat.replace(',', '.')) if isinstance(str_lat, str) else str_lat

            # Списки азимутов и углов по горизонтале
            filtered_df = df_anten[
                (df_anten['CELL_LON_COORD'] == str_long) &
                (df_anten['CELL_LAT_COORD'] == str_lat)
                ]
            ls_antenna_id = list(filtered_df['ANTENNA_ID'])
            dc_anten_2 = dict()
            ls_antenn = []
            for id_ant, antenna_id in enumerate(ls_antenna_id):
                dc_anten_2['antenna_id'] = antenna_id
                dc_anten_2['ANTENNA_NAME'] = filtered_df['ANTENNA_NAME'].iloc[id_ant]
                dc_anten_2['coord_long'] = float_long
                dc_anten_2['coord_lat'] = float_lat
                dc_anten_2['AZIMUTH'] = int(filtered_df['AZIMUTH'].iloc[id_ant])
                dc_anten_2['HBW'] = int(filtered_df['HBW'].iloc[id_ant])

                power = filtered_df['POWER'].iloc[id_ant]
                float(power.replace(',', '.')) if isinstance(power, str) else power
                dc_anten_2['POWER'] = power

                height = filtered_df['HEIGHT'].iloc[id_ant]
                float(height.replace(',', '.')) if isinstance(height, str) else height
                dc_anten_2['HEIGHT'] = height

                ls_antenn.append(copy.copy(dc_anten_2))
            # Итоговый словарь
            dc_anten[f'Tower_{i}'] = ls_antenn

        # print(dc_anten)
        return dc_anten

    @property
    def get_antenn_ret(self) -> List:
        """ Возвращает список антенн с признаком RET = 1

        """
        ls_id_antenn_ret = [id for id, dc_val in self.dc_anten_v2.items() if dc_val['ret']]
        return ls_id_antenn_ret

    def call_distance(self, **kwargs):
        """Расчёт перпендикулярного расстояние от антенны до ячейки и угол между векторами и
        расстояние между антенной и точкой (в метрах)

        :return: Словарь результатов {'ds': расстояние между антенной и точкой,
                                    'perp': Перпендикулярное расстояния до антенны,
                                    'angle_between': Угол между направлением антенны и направлением на ячейку}
        """
        azimuth = int(kwargs['azimuth'])  # !!! int() -- для отладки. Угол направления антенны (в градусах от севера)
        center_lng = kwargs['center_lng']  # Долгота ячейки
        center_lat = kwargs['center_lat']  # Широта ячейки
        anten_lng = kwargs['anten_lng']  # Долгота антенны
        anten_lat = kwargs['anten_lat']  # Широта антенны

        # Расстояние между антенной и точкой (в метрах)
        ds = int(distance((center_lat, center_lng), (anten_lat, anten_lng)).meters)

        # Преобразуем азимут в радианы
        try:
            antenna_angle_rad = np.radians(azimuth)
        except Exception as e:
            print(f'Error {azimuth=},  {e}')
            return None

        # Вектор направления антенны (ось Y соответствует северу)
        antenna_vector = np.array([np.sin(antenna_angle_rad), np.cos(antenna_angle_rad)])

        # Прямое расстояние между вышкой и ячейкой
        distance_tower_to_cell = ds

        # Разница по долготе и широте между вышкой и ячейкой
        delta_lat = center_lat - anten_lat
        delta_lon = center_lng - anten_lng

        # Вектор "вышка-ячейка"
        tower_to_cell_vector = np.array([delta_lon, delta_lat])

        # Нормализуем векторы для корректного вычисления угла
        normalized_tower_to_cell = tower_to_cell_vector / np.linalg.norm(tower_to_cell_vector)
        normalized_antenna_vector = antenna_vector / np.linalg.norm(antenna_vector)

        # Скалярное произведение для нахождения косинуса угла между векторами
        dot_product = np.dot(normalized_tower_to_cell, normalized_antenna_vector)

        # Угол между направлением антенны и направлением на ячейку (в радианах)
        angle_between = np.arccos(np.clip(dot_product, -1.0, 1.0))

        # Если угол больше 90 градусов, значит ячейка находится за антенной
        angle_0 = int(np.degrees(angle_between))
        if angle_0 > 90:
            # print(f"Ячейка находится за антенной. {angle_0=}")
            dc_result = {'ds': ds,
                         'perp': None,
                         'angle_between': angle_between,
                         'angle_bet_grad': np.rad2deg(angle_between)}
            return dc_result
        # print("Ячейка находится перед антенной.")

        # Перпендикулярное расстояние от ячейки до направления антенны
        perpendicular_distance = int(distance_tower_to_cell * np.sin(angle_between))

        # Выводим результаты
        dc_result = {'ds': ds,
                     'perp': perpendicular_distance,
                     'angle_between': angle_between,
                     'angle_bet_grad': np.rad2deg(angle_between), }
        return dc_result

    def calculate_a_hm(self, frequency_mhz=1800, h_m=1.8):
        """
        Расчет корректирующего коэффициента a(h_m) для высоты антенны мобильной станции дБ

        :param frequency_mhz: Диапазон рабочих частот
        :param h_m: Высота подъёма антенны МС
               по умолчанию = 1,8 м.
        """
        log_f = math.log10(frequency_mhz)
        a_hm = (1.1 * log_f - 0.7) * h_m - (1.56 * log_f - 0.8)
        return a_hm

    def calculate_power(self, P_tx_watts, h_b, d_km, A_angle_dB,
                        A_angle_v_dB: float,
                        frequency_mhz: int = 1800,
                        C: int = 0):
        """ Вычисление мощности в точке приёма дБм

        :param P_tx_watts: Мощность сигнала антенны, Вт
        :param A_angle_dB: Дополнительное затухание из-за угла, дБ
        :param A_angle_v_dB: Дополнительное затухание из-за угла наклона, дБ
        :param frequency_mhz: Диапазон рабочих частот
        :param h_b: Высота антенны базовой станции, м
        :param d_km: Расстояние до точки, км
        :param a_hm: корректирующего коэффициент для высоты антенны мобильной станции
        :param C: Корректирующий коэффициент для городской среды, дБ
                Для COST 231-Hata
                0 - открытое пространство, пригород, село; 3 - город
        """
        if isinstance(P_tx_watts, str):
            P_tx_watts = float(P_tx_watts.replace(',', '.'))

        if isinstance(h_b, str):
            h_b = float(h_b.replace(',', '.'))

        log_f = math.log10(frequency_mhz)
        log_hb = math.log10(h_b)

        log_d = math.log10(d_km)
        A = 46.3  # Постоянная составляющая потерь
        B = 33.9  # Коэффициент коррекции частоты

        # Конвертация мощности передатчика из Вт в дБм
        P_tx_mW = P_tx_watts * 1000  # Вт в мВт
        P_tx_dBm = 10 * math.log10(P_tx_mW)

        los = A + B * log_f - 13.82 * log_hb - self.a_hm + abs((44.9 - 6.55 * log_hb)) * log_d + C
        P = P_tx_dBm - los + A_angle_dB + A_angle_v_dB
        return P

    def calculate_power_id(self, center_lng, center_lat, antenn_id,
                           new_angle: int = 400,
                           frequency_mhz: int = 1800,
                           C: int = 0,
                           uses_tilt: bool = False):
        """ Вычисление мощности в точке приёма дБм для антенны с данным id

        :param frequency_mhz: Диапазон рабочих частот
        :param C: Корректирующий коэффициент для городской среды, дБ
                Для COST 231-Hata
                0 - открытое пространство, пригород, село; 3 - город
        :param uses_tilt: Использовать данные угла наклона антенны
        """
        dc_antenn_prm = self.get_nearest_antenna_v3(center_lng=center_lng, center_lat=center_lat, antenna_id=antenn_id)
        # dc_antenn_prm = self.dc_anten_v2[antenn_id]

        P_tx_watts = dc_antenn_prm['power']  # Мощность сигнала антенны, Вт
        h_b = dc_antenn_prm['height']  # Высота антенны базовой станции, м
        d_m = max(0.01, dc_antenn_prm['ds'])
        d_km = d_m / 1000  # Расстояние до точки, км
        tilt = dc_antenn_prm['tilt'] # Угол наклона антенны

        if new_angle == 400:
            angle = dc_antenn_prm['angle_bet_grad']  # Угол между азимутом и вектором
        else:
            angle = new_angle
        # Дополнительное затухание из-за угла (по горизонтали), дБ
        A_angle_dB = self.get_angle_rotate(antenn_id=antenn_id,
                                           angle=angle)
        # Дополнительное затухание из-за угла (по ветикали), дБ
        if uses_tilt:
            A_angle_v_dB = self.get_angle_tilt(antenn_id=antenn_id,
                                               tilt=tilt)
        else:
            A_angle_v_dB = 0

        log_f = math.log10(frequency_mhz)
        try:
            log_hb = math.log10(h_b)
            log_d = math.log10(d_km)
        except Exception as e:
            print(f'{dc_antenn_prm=}, Error: {e}')
        A = 46.3  # Постоянная составляющая потерь
        B = 33.9  # Коэффициент коррекции частоты

        # Конвертация мощности передатчика из Вт в дБм
        P_tx_mW = P_tx_watts * 1000  # Вт в мВт
        P_tx_dBm = 10 * math.log10(P_tx_mW)

        # a_hm: корректирующего коэффициент для высоты антенны мобильной станции
        los = A + B * log_f - 13.82 * log_hb - self.a_hm + abs((44.9 - 6.55 * log_hb)) * log_d + C
        P = P_tx_dBm - los + A_angle_dB + A_angle_v_dB
        return int(P)

    def get_max_power_id(self, latitude, longitude, all: bool = False):
        """ Нахождение антенны с максимальным сигналом / или всех отсортированных по
        уровню сигнала

        :param latitude: Координаты точки измерения (широта)
        :param longitude: Координаты точки измерения (долгота)
        :param all: Выгружать всех отсортированных по уровню сигнала
        :return:
        """
        # Получение данных всех антенны
        nearest_antenna = self.get_antenns_params(center_lng=longitude, center_lat=latitude)
        dc_power_antenn = {}
        # Для всех антенн находим мохность сигнала в точке приёма
        for key, dc_val in nearest_antenna.items():
            angle = dc_val['angle_bet_grad']
            tilt = dc_val['tilt']
            A_angle_dB = self.get_angle_rotate(antenn_id=key, angle=angle)
            # A_angle_v_dB = self.get_angle_tilt(antenn_id=key, tilt=tilt)
            A_angle_v_dB = 0

            d_m = max(0.01, dc_val['ds'])
            try:
                p = self.calculate_power(P_tx_watts=dc_val['power'],
                                         h_b=dc_val['height'],
                                         d_km=d_m / 1000,
                                         A_angle_dB=A_angle_dB,
                                         A_angle_v_dB=A_angle_v_dB)
            except ValueError as e:
                print(f"Ошибка при расчёте мощности: {e}, данные: {dc_val}")
                p = None
            dc_power_antenn[key] = p
        # Выбираем антенну с максимальным сигналом
        dc_anten = dict(sorted(dc_power_antenn.items(), key=lambda item: item[1]))
        if all:
            id_antenn_max = list(dc_anten)[::-1]
        else:
            id_antenn_max = list(dc_anten)[::-1][0]
        # print(id_antenn_max)
        return id_antenn_max

    def get_angle_rotate(self, antenn_id, angle):
        """ Находит дополнительное затухание из-за угла, дБ по заданному углу

        :param antenn_id: id антенны
        :param angle: угол между азимутом и вектором направления на точку
        """
        angle = 179 if angle == 180 else angle
        antenna_name = self.dc_anten_v2[antenn_id]['ANTENNA_NAME']
        try:
            signal_atten = self.antennas_gains[str(antenna_name)][str(int(angle))]
        except Exception as e:
            print(f'{antenna_name=}, {angle=}, Error: {e}')
            signal_atten = 0
        return signal_atten

    def get_angle_tilt(self, antenn_id, tilt):
        """ Находит дополнительное затухание из-за угла наклона, дБ по заданному углу

        :param antenn_id: id антенны
        :param tilt: угол наклона антенны
        """

        antenna_name = self.dc_anten_v2[antenn_id]['ANTENNA_NAME']
        signal_atten = self.antennas_gains_tilt[str(antenna_name)][str(int(tilt))]
        return signal_atten

    def wat_2_db(self, wat):
        db_fl = float(10 * np.log10(wat))
        db = round(db_fl, 2)
        return db

    def dbm_2_wat(self, dbm):
        return 10 ** ((dbm - 30) / 10)


if __name__ == '__main__':
    pass
    # tuner = AntennaFeederTuner()
    # print(tuner.df_max_rsrp_from_squar.head())
    # tuner.create_dataset()
    # print(list(tuner.antennas_gains))
    # print(tuner.antennas_gains[739623])
