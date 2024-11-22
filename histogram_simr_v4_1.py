import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map
from itertools import chain

from core.af_tuner_module import AntennaFeederTuner

def calculate_sinr_for_chunk(chunk, tuner, id_antenn_tilt, tilt:int = 400):
    """ Рассчитываем SINR для чанка данных с заданным наклоном.
        Аргументы приходят в виде кортежа (chunk, tuner, tilt, id_antenn_tilt)
    """
    ls_sinr_db = []
    if tilt != 400:
        # Обновляем наклон для данной антенны
        tuner.dc_anten_v2[id_antenn_tilt].update({'tilt': tilt})

    for row in chunk.itertuples():
        ls_id_antenn = tuner.get_max_power_id(latitude=row.latitude, longitude=row.longitude, all=True)
        # учитывать квадрат в котором данная антенна главная
        # if ls_id_antenn[0] != id_antenn_tilt:
        #     continue
        sinr_db = tuner.call_sinr_v2(center_lng=row.longitude,
                                      center_lat=row.latitude,
                                      ls_id_antenn=ls_id_antenn,
                                      uses_tilt=True)
        ls_sinr_db.append(sinr_db)
    return ls_sinr_db

def call_sinr():
    """ Наклоняем антенну и вычисляем SINR для каждой точки из to_histogram_5x5_1800.csv
        Результат в histogram_sinr_id_{id_antenn_tilt}_ret.csv
        Поля: id_squar, sinr_db
    """
    tuner = AntennaFeederTuner('LTE1800')
    # [83, 73, 6, 16, 98, 5]
    # Список антенн с RET -- 16, 73
    ls_id_antenn_tilt = tuner.get_antenn_ret
    df_csv: pd.DataFrame = pd.read_csv('to_histogram_5x5_1800_10km.csv')
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
        # for tilt in tqdm(range(-65, 4, 4), desc="Tilts"):  # Проходим по всем углам наклона
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

def get_opt_tilt(file_name: str):
    # Пороговое значение
    val_p = 1
    # Словарь для хранения результатов
    results = {}
    df = pd.read_csv(file_name)

    # Перебираем колонки tilt_0 .. tilt_10
    for col in [f"sinr_tilt_{i}" for i in range(-1, 7, 1)]:
        # Условие 1: etl < val_p, tilt > val_p -- уровень сигнал стал ВЫШЕ прога
        count_condition_1 = df[(df["sinr_etl"] < val_p) & (df[col] > val_p)].shape[0]
        # Условие 2: etl > val_p, tilt > etl -- уровень сигнала был ВЫШЕ прога и ещё улучшился
        count_condition_2 = df[(df["sinr_etl"] > val_p) & (df[col] > df["sinr_etl"])].shape[0]
        # Условие 3: etl > val_p, tilt < val_p -- уровень сигнала стал меньше ПОРОГА, сигнал ухудшился
        count_condition_3 = df[(df["sinr_etl"] > val_p) & (df[col] < df["sinr_etl"])].shape[0]
        # Сохраняем в словарь
        results[col] = (count_condition_1, count_condition_2, count_condition_3)

    sorted_results = dict(
        sorted(results.items(), key=lambda item: (-item[1][2], -item[1][0]), reverse=True)
    )
    return sorted_results


if __name__ == '__main__':
    # call_sinr()
    s = get_opt_tilt('histogram_sinr_id_1_ret.csv')
    print(s)