import pandas as pd
from typing import List, Dict
import geopandas as gpd
from dask.array import arctan
from shapely.geometry import Polygon
from geopy.distance import geodesic
import math
import json
import numpy as np




class Optimizer:

    def __init__(self):
        self.df_antenn = pd.read_excel('../sourse_data/Тихвин_соты_30.09.2024_1800_RET.xlsx')
        self.df_squar = pd.read_csv('../sourse_data/squar_coord_5.csv')
        with open('../sourse_data/antennas_gains_vertical.dat', 'rb') as f:
            self.antennas_gains_tilt = json.loads(f.read())
        self.antenn_ret = self._ger_antenn_ret

    @property
    def _ger_antenn_ret(self) -> List[int]:
        """ Возвращает список антенн с RET

        """
        df_antenn = self.df_antenn[self.df_antenn['RET'] == 1]
        antenn_ret = df_antenn['cellname'].unique().tolist()
        return antenn_ret


    def find_square_center(self, id_squar):
        """Расчёт координат центра квадрата """

        latitude_1, longitude_1, latitude_2, longitude_2 = self._get_coord_square(id_squar)

        # Создаем полигон (квадрат) из двух углов
        polygon = Polygon([(longitude_1, latitude_1), (longitude_2, latitude_1),
                           (longitude_2, latitude_2), (longitude_1, latitude_2)])

        # Преобразуем в GeoSeries
        gdf = gpd.GeoSeries([polygon])

        # Находим центр полигона
        centroid = gdf.geometry.centroid.iloc[0]

        # Возвращаем координаты центра
        latitude, longitude = centroid.y, centroid.x
        return latitude, longitude

    def _get_coord_square(self, id_squar):
        """ Возвращает координаты углов квадрата

        """
        squar = self.df_squar[self.df_squar['id_squar'] == id_squar]
        if not squar.empty:
            latitude_1 = squar.iloc[0]['latitude_1']
            longitude_1 = squar.iloc[0]['longitude_1']
            latitude_2 = squar.iloc[0]['latitude_2']
            longitude_2 = squar.iloc[0]['longitude_2']
            return latitude_1, longitude_1, latitude_2, longitude_2
        else:
            return None

    def get_antenna_coordinates(self, id_antenn):
        """
        Возвращает координаты антенны.

        Параметры:
            id_antenn (int): Идентификатор антенны.

        Возвращает:
            tuple: Широта и долгота антенны (latitude, longitude).

        Исключения:
            ValueError: Если антенна с заданным ID не найдена.
        """
        df_antenn = self.df_antenn.loc[self.df_antenn['cellname'] == id_antenn, ['CELL_LAT_COORD', 'CELL_LON_COORD']]
        if df_antenn.empty:
            raise ValueError(f"Аntenna ID {id_antenn} не найден.")
        latitude = df_antenn.iloc[0]['CELL_LAT_COORD']
        latitude = float(latitude.replace(',','.')) if isinstance(latitude, str) else latitude
        longitude = df_antenn.iloc[0]['CELL_LON_COORD']
        longitude = float(longitude.replace(',', '.')) if isinstance(longitude, str) else longitude
        return latitude, longitude

    def get_tilt_antenn(self, cellname) -> int:
        """ Возвращает суммарный угол наклона антенны"""
        df_antenn = self.df_antenn.loc[self.df_antenn['cellname'] == cellname, ['TILT']]
        tilt = int(df_antenn.iloc[0]['TILT'])
        return tilt

    def get_height_antenn(self, cellname) -> float:
        """ Возвращает высоту антенны"""
        df_antenn = self.df_antenn.loc[self.df_antenn['cellname'] == cellname, ['HEIGHT']]
        height = df_antenn.iloc[0]['HEIGHT']
        height = float(height.replace(',','.')) if isinstance(height, str) else height
        return height


    def get_distance(self, id_squar, cellname):
        """ Рассчитывает расстояние от центра квадрата до антенны в метрах

        """
        latitude_antenn, longitude_antenn = self.get_antenna_coordinates(cellname)
        latitude_squar, longitude_squar = self.find_square_center(id_squar)
        # Рассчитываем расстояние
        distance = geodesic((latitude_squar, longitude_squar), (latitude_antenn, longitude_antenn)).meters
        distance = round(distance, 1)

        return distance

    def call_tilt(self, id_squar: int, cellname: str, debug: bool = False):
        """ Вычисляет величины угла между главным лучом антенны и
        вектором направления на точку (в грудусах)

        :param id_squar: id квадрата
        :param cellname: id антенны
        :param debug: Режим отладки
        :return: Вычисленное значение угла между главным лучом антенны и вектором направления
        """
        distance = self.get_distance(id_squar, cellname)
        # Суммарный угол наклона антенны
        tilt = self.get_tilt_antenn(cellname)
        height = self.get_height_antenn(cellname)
        # Угол между горизонтом и вектором направление на точку
        angle_radians = math.atan((height - 1.7) / distance)
        az_angle = math.degrees(angle_radians)
        angle = round(az_angle - tilt)
        if debug:
            print("*** Вычисление угла между главным лучом антенны и вектором направления ***")
            print(f'{tilt=}, {az_angle=}, {angle=}, {height=}, {distance=}')
        return angle

    def get_signal_weakening(self, cellname, tilt):
        """ Возвращает величину ослабевания сигнала антенны при данном угле наклона

        """
        df_antenn = self.df_antenn.loc[self.df_antenn['cellname'] == cellname, ['ANTENNA_NAME']]
        antenna_name = df_antenn.iloc[0]['ANTENNA_NAME']
        signal_weakening = self.antennas_gains_tilt[str(antenna_name)][str(int(tilt))]
        return signal_weakening

    def wat_2_db(self, wat):
        return 10 * np.log10(wat)

    def dbm_2_mwat(self, dbm):
        return 10 ** (dbm / 10)

    def call_sinr(self, ls_rsrp: List, debug: bool = False):
        """ Вычисление SINR для списка RSRP.

        :param ls_rsrp: Список усреднённых значений RSRP
        :param debug: Режим отладки
        :return: Вычисленное значение SINR
        """
        thermal_noise_watt = self.dbm_2_mwat(-122)
        max_rsrp_dbm = sorted(ls_rsrp)[-1]
        interf_rsrp_dbm = sorted(ls_rsrp)[:-1]
        signal_mwatt = self.dbm_2_mwat(max_rsrp_dbm)

        interf_mwatt = sum(map(self.dbm_2_mwat, interf_rsrp_dbm))
        sinr_raz = signal_mwatt / (interf_mwatt + thermal_noise_watt)
        # При интерференции == 0, SINR = 14
        sinr = self.wat_2_db(sinr_raz) if interf_rsrp_dbm else 14
        sinr = 25 if sinr > 25 else sinr
        if debug:
            print('*** Вычисление SINR для списка RSRP ***')
            print(f'{ls_rsrp=}')
            print(f'{max_rsrp_dbm=}, {interf_rsrp_dbm=}')
            print(f'{signal_mwatt=}, {interf_mwatt=}, {thermal_noise_watt=}, {sinr_raz=}, {sinr=}')
        return round(sinr, 2)

    def calc_rsrp_tilt(self, dc_antenns: Dict, cellname: str, add_tilt: int):
        """ Расчёт rsrp при угле наклона антенны

        :param dc_antenns: словарь параметров антенн {'TIHVINSVIR3LM': {'rsrp': -93.0, 'cn_ngb': 2, 'delta_tilt': -4}, ...}
        :param add_tilt: величина на которую изменяем угол наклона
        :param cellname: id антенны
        :return: rsrp при угле наклона антенны
        """
        # угол между главным лучом антенны и вектором направление
        delta_tilt = dc_antenns[cellname]['delta_tilt']
        # новый угол между главным лучом антенны и вектором направление
        new_tilt = delta_tilt - add_tilt
        # ослабевание сигнала
        old_signal_weakening = self.get_signal_weakening(cellname=cellname, tilt=delta_tilt)
        signal_weakening = self.get_signal_weakening(cellname=cellname, tilt=new_tilt)

        old_rsrp = dc_antenns[cellname]['rsrp']
        delta_signal = old_signal_weakening - signal_weakening
        new_rsrp = old_rsrp - delta_signal
        return new_rsrp



    def calc_sinr_tilt(self, dc_antenns: Dict, cellname: str, add_tilt: int, debug:bool = False):
        """ Вычисление SINR при увеличении наклона вычисляем новый угол между главным лучом антенны и
            вектором направление на точку (в грудусах)

        delta_tilt -- величины угла между главным лучом антенны и вектором направление на точку (в грудусах)
        (-)delta_tilt -- вектор выше луча, (+)delta_tilt -- вектор ниже луча
        (-)add_tilt -- поднимаем антенну, (+)add_tilt -- опускаем антенну

        :param dc_antenns: словарь параметров антенн {'TIHVINSVIR3LM': {'rsrp': -93.0, 'cn_ngb': 2, 'delta_tilt': -4}, ...}
        :param add_tilt: величина на которую изменяем угол наклона
        :param cellname: id антенны
        :param delta_tilt: угол между главным лучом антенны и вектором направление
        """
        delta_tilt = dc_antenns[cellname]['delta_tilt']
        new_tilt = delta_tilt - add_tilt
        # ослабевание сигнала
        old_signal_weakening = self.get_signal_weakening(cellname=cellname, tilt=delta_tilt)
        signal_weakening = self.get_signal_weakening(cellname=cellname, tilt=new_tilt)
        delta_signal = old_signal_weakening - signal_weakening

        old_rsrp = dc_antenns[cellname]['rsrp']
        dc_antenns.update({cellname: {'rsrp': old_rsrp - delta_signal}})
        new_rsrp = dc_antenns[cellname]["rsrp"]
        ls_rsrp = [val['rsrp'] for val in dc_antenns.values()]

        if debug:
            print('*** Вычисление SINR при увеличении наклона вычисляем новый угол ***')
            tilt = self.get_tilt_antenn(cellname)
            print(
                f'{cellname=}, {tilt=}, {add_tilt=}, {delta_tilt=}, {new_tilt=}, {old_signal_weakening=}, {signal_weakening=}, {delta_signal=}')
            print(f'{old_rsrp=}, {new_rsrp=}, {ls_rsrp=}')

        sinr = self.call_sinr(ls_rsrp, debug)
        return sinr


    def calc_new_sinr(self, row, cellname, add_tilt, debug:bool = False):
        """ Вычисление SINR при увеличении наклона вычисляем новый угол между главным лучом антенны и
            вектором направление на точку (в грудусах)

        delta_tilt -- величины угла между главным лучом антенны и вектором направление на точку (в грудусах)
        (-)delta_tilt -- вектор выше луча, (+)delta_tilt -- вектор ниже луча
        (-)add_tilt -- поднимаем антенну, (+)add_tilt -- опускаем антенну

        :param add_tilt: величина на которую изменяем угол наклона
        :param cellname: id антенны
        :param row: данные для расчёта
        """
        # угол между главным лучом антенны и вектором направление
        delta_tilt = row['delta_tilt']
        # новый угол между главным лучом антенны и вектором направление
        new_tilt = delta_tilt - add_tilt
        # ослабевание сигнала
        old_signal_weakening = self.get_signal_weakening(cellname=cellname, tilt=delta_tilt)
        signal_weakening = self.get_signal_weakening(cellname=cellname, tilt=new_tilt)
        delta_signal = old_signal_weakening - signal_weakening

        dc_antenns = eval(row['antenn'])
        old_rsrp = dc_antenns[cellname]['rsrp']
        dc_antenns.update({cellname: {'rsrp': old_rsrp - delta_signal}})
        new_rsrp = dc_antenns[cellname]["rsrp"]
        if debug:
            print('*** Вычисление SINR при увеличении наклона вычисляем новый угол ***')
            tilt = self.get_tilt_antenn(cellname)
            print(
                f'{tilt=}, {add_tilt=}, {delta_tilt=}, {new_tilt=}, {old_signal_weakening=}, {signal_weakening=}, {delta_signal=}')
            print(f'{old_rsrp=}, {new_rsrp=}')

        ls_rsrp = [val['rsrp'] for val in dc_antenns.values()]
        if new_rsrp < max(ls_rsrp):
            sinr = -10
        else:
            sinr = self.call_sinr(ls_rsrp, debug)

        return sinr



if __name__ == '__main__':
    optimizer = Optimizer()
    # distance = optimizer.get_distance(id_squar=63553, id_antenn=475016)
    # print(distance)

    # print(optimizer.get_tilt_antenn(475016))

    # print(optimizer.get_height_antenn(475016))
    # print(optimizer.get_height_antenn(470189))

    # res = optimizer.call_tilt(id_squar=63553, id_antenn=475016)
    # res = optimizer.call_tilt(id_squar=63553, id_antenn=470189)

    res = optimizer.call_sinr([-117.62, -117.0, -122.0, -117.0, -120.0, -122.0, -122.0])

    print(res)
