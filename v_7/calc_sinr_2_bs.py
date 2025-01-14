"""
Многопроцессорное вычисление средневзвешенного SINR для
всех комбинации углов наклона для ДВУХ базовых станций (6 антенн, 5 варианта угла наклона)
"""

import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import json
from itertools import product

from v_7.optim_v7 import Optimizer

optimizer = Optimizer()


def create_combinations(df_cn_neighbour):
    # Set уникальных антенн
    # unicue_antenn = set()
    # for row in df_cn_neighbour.itertuples():
    #     ls_antenn = list(eval(row.antenn))
    #     unicue_antenn.update(ls_antenn)
    # print(unicue_antenn)
    # unicue_antenn = list(unicue_antenn)

    # Из 8 антенн исключаем антенны: 'TIKHVINDK3LM' азимут 260, 'TIHVINSVIR1LM' азимут 45
    unicue_antenn = ['TIKHVINDK1LM', 'TIHVINSVIR3SLM', 'TIHVINSVIR2LM', 'TIHVINSVIR7LM', 'TIHVINSVIR3LM', 'TIKHVINDK1SLM']

    # Возможные углы наклона
    angles = [-2, -1, 1, 2, 3]

    # Генерация всех комбинаций наклонов
    all_combinations = [
        dict(zip(unicue_antenn, combination))
        for combination in product(angles, repeat=len(unicue_antenn))
    ]

    print(f'Кол-во комбинаций: {len(all_combinations)}')
    return all_combinations


def calculate_sinr(row, combination, optimizer):
    """ Вычисление SINR для комбинации углов наклона для конкретного квадрата

    """
    dc_antenn = eval(row.antenn)
    dc_antenn_copy = dc_antenn.copy()
    # print(f'{dc_antenn=}')
    fl_calc = False
    ls_old_rsrp =  [val['rsrp'] for val in dc_antenn.values()]
    ls_new_rsrp = []

    for antenn, angles in combination.items():
        if antenn in dc_antenn:
            fl_calc = True
            rsrp = optimizer.calc_rsrp_tilt(dc_antenns=dc_antenn_copy, cellname=antenn, add_tilt=angles)
            dc_antenn_copy[antenn]['rsrp'] = rsrp
            # print(f'{antenn=}, {angles=}, {rsrp=}')

    # Вычисление SINR
    if fl_calc:
        # Если антенна из комбинации есть в этом квадрате
        ls_new_rsrp =  [val['rsrp'] for val in dc_antenn.values()]
        sinr = optimizer.call_sinr(ls_rsrp=ls_new_rsrp, debug=False)
    else:
        sinr = row.sinr
    # print(f'{ls_old_rsrp=}, {ls_new_rsrp=}, {fl_calc=}, {sinr=}')
    return sinr


def calculate_combin_sinr(combination, df_cn_neighbour):
    """ Вычисление средневзвешенного SINR для комбинации углов наклона

    :param combination: комбинация углов наклона
                        {'TIKHVINDK1LM': -2, 'TIHVINSVIR3SLM': -2, 'TIHVINSVIR2LM': -2, 'TIHVINSVIR7LM': -2, 'TIHVINSVIR3LM': -2, 'TIKHVINDK1SLM': 2}
    :param df_cn_neighbour: df c полями: id_squar, antenn, cn_neighbour, sinr (файл v_7/add_etl_sinr_step_5)
    :return: sv_sinr - средневзвешенный SINR;
             combination -- комбинация углов наклона
    """
    try:
        series_sinr = df_cn_neighbour.apply(
            lambda row: calculate_sinr(row, combination=combination, optimizer=optimizer),
            axis=1
        )
        sm_sinr = (df_cn_neighbour['cn_neighbour'] * series_sinr).sum()
        cn_negborn_0 = df_cn_neighbour['cn_neighbour'].sum()
        sv_sinr = sm_sinr / cn_negborn_0
        return sv_sinr, combination
    except Exception as e:
        return f"Error: {e}", combination


def main():
    """ Многопроцессорный запуск вычисление средневзвешенного SINR для
    всех комбинации углов наклона

    """
    is_header_written = False
    df_cn_neighbour = pd.read_csv('../v_7/add_etl_sinr_step_5')
    # df_cn_neighbour = df_cn_neighbour.iloc[:50]
    all_combinations = create_combinations(df_cn_neighbour)
    process_partial = partial(calculate_combin_sinr, df_cn_neighbour=df_cn_neighbour)

    with open('../v_7/final_step_7', 'w', newline='', encoding='utf-8') as f:
        with Pool(processes=cpu_count()) as pool:
                for idx, result in enumerate(pool.imap_unordered(process_partial, all_combinations)):
                    print(f"Result {idx}: {result}")
                    df_result = pd.DataFrame(
                        [{'sv_sinr': result[0], 'combination': json.dumps(result[1])}])
                    df_result.to_csv(f, mode='a', header=not is_header_written, index=False)
                    is_header_written = True


if __name__ == '__main__':
    main()
