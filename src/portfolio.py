from typing import List

import numpy as np


class PortfolioFormer:
    def __init__(
        self,
        train_scale: int,
        look_back_window: int,
        shortest_seq: int,
        future_window: int,
        stock_conf
    ) -> None:
        self.train_scale = train_scale
        self.look_back_window = look_back_window
        self.shortest_seq = shortest_seq
        self.future_window = future_window
        self.stock_conf = stock_conf

    def get_returns(self, stock_dict: dict) -> List[np.ndarray]:
        return_matrix_list = []

        for i in range(
            self.train_scale + self.look_back_window + 1,
            self.shortest_seq,
            self.future_window
        ):
            if i + self.future_window > self.shortest_seq:
                break

            return_matrix = np.zeros((self.future_window, len(self.stock_conf.for_expr)))

            for j, (_, stock_df) in enumerate(stock_dict.items()):
                return_matrix[:, j] = stock_df.iloc[i: i + self.future_window]["ret"].values

            return_matrix_list.append(return_matrix)

        return return_matrix_list

    def form_portfolio(
        self,
        y_preds: List[np.ndarray],
        stock_dict: dict,
        long_num: int,
        short_num: int,
    ):
        sorted_pred_indices = []

        for i in range(len(y_preds)):
            sorted_pred_indices.append(np.argsort(y_preds[i]))

        return_matrix_list = self.get_returns(stock_dict)

        n_days = len(return_matrix_list)
        long_returns = np.zeros((n_days, self.future_window, long_num))
        short_returns = np.zeros((n_days, self.future_window, short_num))
        bench_returns = np.zeros((n_days, self.future_window, len(self.stock_conf.for_expr)))

        for day in range(n_days):
            top_k_indices = sorted_pred_indices[day][-long_num:]
            bottom_k_indices = sorted_pred_indices[day][:short_num]
            bench_indices = sorted_pred_indices[day]

            long_returns[day] = return_matrix_list[day][:, top_k_indices]
            short_returns[day] = return_matrix_list[day][:, bottom_k_indices]
            bench_returns[day] = return_matrix_list[day][:, bench_indices]

        long_returns_flat = np.concatenate(long_returns, axis=0).mean(axis=1)
        short_returns_flat = -1 * np.concatenate(short_returns, axis=0).mean(axis=1)
        bench_returns_flat = np.concatenate(bench_returns, axis=0).mean(axis=1)
        portfolio_returns_flat = (long_returns_flat + short_returns_flat) / 2

        return (
            long_returns_flat,
            short_returns_flat,
            bench_returns_flat,
            portfolio_returns_flat,
        )