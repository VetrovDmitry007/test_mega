import cProfile
import pstats
from io import StringIO
from typing import List, Dict, Tuple

import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial
from time import perf_counter

from tqdm.contrib import itertools
from tqdm.contrib.concurrent import process_map
from itertools import chain

from core.af_tuner_module import AntennaFeederTuner

def calculate_sinr_for_chunk(chunk, tuner, id_antenn_tilt:int = 0, tilt:int = 400):
    """ Рассчитываем SINR для чанка данных с заданным наклоном.
        Аргументы приходят в виде кортежа (chunk, tuner, tilt, id_antenn_tilt)
    """
    ls_sinr_db = []
    if tilt != 400:
        # Обновляем наклон для данной антенны
        tuner.dc_anten_v2[id_antenn_tilt].update({'tilt': tilt})

    for row in chunk.itertuples():
        dc_power_antenn = tuner.get_max_power_id(latitude=row.latitude, longitude=row.longitude, all=True)
        # учитывать квадрат в котором данная антенна главная
        # if ls_id_antenn[0] != id_antenn_tilt:
        #     continue
        sinr_db = tuner.call_sinr_v2(center_lng=row.longitude,
                                      center_lat=row.latitude,
                                      dc_power_antenn=dc_power_antenn,
                                      uses_tilt=True)
        ls_sinr_db.append(sinr_db)
    return ls_sinr_db

def calculate_sinr_for_chunk_v2(chunk, tuner, id_antenn_tilt:int = 0, tilt:int = 400):
    """ Рассчитываем SINR для чанка данных с заданным наклоном.
        Аргументы приходят в виде кортежа (chunk, tuner, tilt, id_antenn_tilt)
    """
    ls_sinr_db = []
    if tilt != 400:
        # Обновляем наклон для данной антенны
        tuner.dc_anten_v2[id_antenn_tilt].update({'tilt': tilt})

    for row in chunk.itertuples():
        dc_power_antenn = tuner.get_max_power_id(latitude=row.latitude, longitude=row.longitude, all=True)
        sinr_db = tuner.call_sinr_v2(dc_power_antenn)
        ls_sinr_db.append((row.id_squar, sinr_db))
        # print(f'latitude={row.latitude}, longitude={row.longitude}, {sinr_db=}')
    return ls_sinr_db

def calculate_sinr_one():
    """ Рассчитываем SINR для чанка данных с заданным наклоном.
        Аргументы приходят в виде кортежа (chunk, tuner, tilt, id_antenn_tilt)
    """
    ls_sinr_db = []
    tuner = AntennaFeederTuner('LTE1800')
    df_csv: pd.DataFrame = pd.read_csv('to_histogram_5x5_1800_10km.csv')
    df_csv = df_csv.loc[0:1000]

    t_1 = perf_counter()
    for row in tqdm(df_csv.itertuples()):
        dc_power_antenn = tuner.get_max_power_id(latitude=row.latitude, longitude=row.longitude, all=True)
        # print(perf_counter() - t_1)
        # t_1 = perf_counter()
        sinr_db = tuner.call_sinr_v2(center_lng=row.longitude,
                                      center_lat=row.latitude,
                                      dc_power_antenn=dc_power_antenn,
                                      uses_tilt=True)
        ls_sinr_db.append(sinr_db)
    print(perf_counter() - t_1)
    return ls_sinr_db


def call_sinr():
    """ Наклоняем антенну и вычисляем SINR для каждой точки из to_histogram_5x5_1800.csv
        Результат в histogram_sinr_id_{id_antenn_tilt}_ret.csv
        Поля: id_squar, sinr_db
    """
    tuner = AntennaFeederTuner('LTE1800')
    ls_id_antenn_tilt = tuner.get_antenn_ret
    df_csv: pd.DataFrame = pd.read_csv('to_histogram_5x5_1800_10km.csv')
    # df_csv = df_csv.loc[0:5000]
    print(f'Read to_histogram_5x5_1800.csv rows: {df_csv.shape}')

    dc_sinr = {}
    # Разделение данных на чанки
    chunk_size = len(df_csv) // cpu_count()
    chunks = [df_csv.iloc[i:i + chunk_size] for i in range(0, len(df_csv), chunk_size)]

    for id_antenn_tilt in ls_id_antenn_tilt:
        print(f'Оптимизация антенны id: {id_antenn_tilt}')
        etl_tilt = tuner.dc_anten_v2[id_antenn_tilt]['tilt']
        start_tilt = etl_tilt - 4
        end_tilt = etl_tilt + 4

        # Вычисление исходного SINR -- sinr_etl
        func = partial(calculate_sinr_for_chunk, tuner=tuner, id_antenn_tilt=id_antenn_tilt)
        results = process_map(func, chunks, max_workers=cpu_count(), desc=f"Processing tilt_etl")
        ls_sinr_db = list(chain(*results))
        dc_sinr.update({f'sinr_etl': ls_sinr_db})

        for tilt in tqdm(range(start_tilt, end_tilt, 1), desc="Tilts"):  # Проходим по всем углам наклона
            # Подготовка функции с использованием partial
            # Зафиксируем параметры tuner, tilt, id_antenn_tilt
            func = partial(calculate_sinr_for_chunk, tuner=tuner, tilt=tilt, id_antenn_tilt=id_antenn_tilt)

            # Использование process_map для многопроцессорной обработки с прогрессом
            results = process_map(func, chunks, max_workers=cpu_count(), desc=f"Processing tilt {tilt}")

            # Сливаем все результаты в один список
            # объединить несколько итерируемых объектов в один
            ls_sinr_db = list(chain(*results))
            dc_sinr.update({f'sinr_tilt_{tilt}': ls_sinr_db})

        # Создаем DataFrame и сохраняем в CSV
        df_sinr = pd.DataFrame.from_dict(dc_sinr)
        df_sinr.to_csv(f'histogram_sinr_id_{id_antenn_tilt}_ret.csv', index=False)
        print(f'Results saved to histogram_sinr_id_{id_antenn_tilt}_ret.csv, {df_sinr.shape=}')


def call_best_sinr():
    """ Для ЛУЧШЕЙ комбинации значений (antenn_id и tilt) вычисляем SINR для каждой точки из to_histogram_5x5_1800.csv
        Результат в opt_sinr_to_histogram.csv

    Лучшая комбинация:
    combin_num='141': [('id_antenn_3', 6), ('id_antenn_6', 2),
                        ('id_antenn_12', 4), ('id_antenn_15', 3),
                        ('id_antenn_24', 3), ('id_antenn_31', 4),
                        ('id_antenn_34', 2), ('id_antenn_35', 2)]
    """
    tuner = AntennaFeederTuner('LTE1800')
    df_csv: pd.DataFrame = pd.read_csv('to_histogram_5x5_1800_10km.csv')
    # df_csv = df_csv.iloc[:10]
    print(f'Read to_histogram_5x5_1800.csv rows: {df_csv.shape}')

    # Разделение данных на чанки
    chunk_size = len(df_csv) // cpu_count()
    chunks = [df_csv.iloc[i:i + chunk_size] for i in range(0, len(df_csv), chunk_size)]

    # !!! Для получения базовых значений SINR, блок комментируется
    # Задание оптимальных углов наклона
    # ls_opt_tilt = [('id_antenn_3', 6), ('id_antenn_6', 2),
    #                     ('id_antenn_12', 4), ('id_antenn_15', 3),
    #                     ('id_antenn_24', 3), ('id_antenn_31', 4),
    #                     ('id_antenn_34', 2), ('id_antenn_35', 2)]
    # for opt_data in ls_opt_tilt:
    #     antenn_id = int(opt_data[0].split('_')[-1])
    #     tilt = opt_data[1]
    #     print(f'{antenn_id=} {tilt=}')
    #     tuner.dc_anten_v2[antenn_id].update({'tilt': tilt})

    # Подготовка функции с использованием partial
    # Зафиксируем параметры tuner, tilt, id_antenn_tilt
    func = partial(calculate_sinr_for_chunk_v2, tuner=tuner)

    # Использование process_map для многопроцессорной обработки с прогрессом
    results = process_map(func, chunks, max_workers=cpu_count(), desc=f"Processing tilt")

    # Сливаем все результаты в один список
    # объединить несколько итерируемых объектов в один
    union_data = list(chain(*results))
    ls_id_squar = [int(row[0]) for row in union_data]
    ls_sinr_db = [row[1] for row in union_data]

    # Создаем DataFrame и сохраняем в CSV
    df_sinr = pd.DataFrame.from_dict({'id_squar': ls_id_squar, 'etl_sinr': ls_sinr_db})
    name_file = 'etl_sinr_to_histogram_2.csv'
    df_sinr.to_csv(name_file, index=False)
    print(f'Results saved to {name_file}, {df_sinr.shape=}')


def get_opt_tilt(file_name: str):
    # Пороговое значение
    val_p = 0
    # Словарь для хранения результатов
    results = {}
    df = pd.read_csv(file_name)

    # Перебираем колонки tilt_0 .. tilt_10
    # В tilt_0 .. tilt_10 значение SINR при различных углах наклона
    for col in [f"sinr_tilt_{i}" for i in range(-1, 7, 1)]:
        # Условие 1: etl < val_p, tilt > val_p -- уровень сигнал стал ВЫШЕ прога
        count_condition_1 = df[(df["sinr_etl"] < val_p) & (df[col] > val_p)].shape[0]
        # Условие 2: etl > val_p, tilt > etl -- уровень сигнала был ВЫШЕ прога и ещё улучшился
        count_condition_2 = df[(df["sinr_etl"] > val_p) & (df[col] > df["sinr_etl"])].shape[0]
        # Условие 3: etl > val_p, tilt < val_p -- уровень сигнала стал меньше ПОРОГА, сигнал ухудшился
        count_condition_3 = df[(df["sinr_etl"] > val_p) & (df[col] < df["sinr_etl"])].shape[0]
        # Условие 4: величина улучшения
        count_condition_4 = df[col].sum() - df["sinr_etl"].sum()
        # Сохраняем в словарь
        results[col] = (count_condition_1, count_condition_2, count_condition_3, count_condition_4)

    sorted_results = dict(
        sorted(results.items(), key=lambda item: (-item[1][2], -item[1][0]), reverse=True)
    )
    return sorted_results


def get_opt_list():
    """ Извлечение результатов оптимизации из итоговых файлов.
    Используется для нахождение оптимального угла для отдельной антенны

    """
    firs_antenn  = 1
    for id in range(firs_antenn, 45, 1):
        try:
            file_opt = f'./from_server/opt_tilt/histogram_sinr_id_{id}_ret.csv'
            dc_opt = get_opt_tilt(file_opt)
            ls_antenn = list(dc_opt)[:4]
            str_res = ''
            for antenn in ls_antenn:
                str_res += f'id_{id} - {antenn}: {dc_opt[antenn]}, '
            # str_res +='\n'
            print(str_res)
        except:
            pass


def call_sinr_comb(antenns: List[Tuple]):
    """
        Для комбинации значений (antenn_id и tilt) вычисляем SINR для каждой точки из to_histogram_5x5_1800.csv
        Результат в histogram_sinr_num_comb_{num_comb}.csv
    """
    tuner = AntennaFeederTuner('LTE1800')
    df_csv: pd.DataFrame = pd.read_csv('to_histogram_5x5_1800_10km.csv')
    # df_csv = df_csv.iloc[:1000]
    print(f'Read to_histogram_5x5_1800.csv rows: {df_csv.shape}')

    # Разделение данных на чанки
    chunk_size = len(df_csv) // cpu_count()
    chunks = [df_csv.iloc[i:i + chunk_size] for i in range(0, len(df_csv), chunk_size)]

    # Обновляем значения углов
    for num_comb, comb_antenn in enumerate(antenns, start=1):
        dc_sinr = {}
        # print(comb_antenn)
        for antenn_id, tilt in comb_antenn:
            antenn_id = int(antenn_id.split('_')[-1])
            tuner.dc_anten_v2[antenn_id].update({'tilt': tilt})

        # Подготовка функции с использованием partial
        # Зафиксируем параметры tuner, tilt, id_antenn_tilt
        func = partial(calculate_sinr_for_chunk, tuner=tuner)

        # Использование process_map для многопроцессорной обработки с прогрессом
        results = process_map(func, chunks, max_workers=cpu_count()-1, desc=f"Processing tilt")

        # Сливаем все результаты в один список
        # объединить несколько итерируемых объектов в один
        ls_sinr_db = list(chain(*results))
        dc_sinr.update({f'{comb_antenn}': ls_sinr_db})

        # Создаем DataFrame и сохраняем в CSV
        df_sinr = pd.DataFrame.from_dict(dc_sinr)
        df_sinr.to_csv(f'histogram_sinr_num_comb_{num_comb}.csv', index=False)
        print(f'Results saved to histogram_sinr_num_comb_{num_comb}.csv, {df_sinr.shape=}')


def run_combin():
    """
    Комбинация id антенн и их углов наклона.
    Пример: (2, 6) -- 2 это базовый угол.
    """
    antennas = [
        {'id_antenn_3': (2, 6)},
        {'id_antenn_6': (2, 6)},
        {'id_antenn_12': (4, 5)},
        {'id_antenn_15': (3, 4)},
        {'id_antenn_24': (2, 3)},
        {'id_antenn_31': (3, 4)},
        {'id_antenn_34': (2, 6)},
        {'id_antenn_35': (2, 6)}
    ]

    # Получение всех возможных комбинаций углов наклона
    angle_combinations = list(itertools.product(*[list(antenna.values())[0] for antenna in antennas]))

    # Формирование списка словарей с нужной структурой
    result = []
    for comb in angle_combinations:
        # Генерация списка кортежей (id антенны, угол)
        antenna_comb = [(list(antennas[i].keys())[0], comb[i]) for i in range(len(antennas))]
        result.append(antenna_comb)

    # Вывод результата
    # print(result)
    return result


if __name__ == '__main__':
    # call_sinr()
    call_best_sinr()

    # combs = run_combin()
    # call_sinr_comb(combs)

    # calculate_sinr_one()

    # s = get_opt_tilt('./from_server/opt_tilt/histogram_sinr_id_1_ret.csv')
    # print(s)
    # get_opt_list()
