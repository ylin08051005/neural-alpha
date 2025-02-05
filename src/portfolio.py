from typing import List, Tuple

import numpy as np


class PortfolioFormer:
    def __init__(
        self,
        train_scale: int,
        look_back_window: int,
        shortest_seq: int,
        future_window: int,
        num_stocks: int,
    ) -> None:
        self.train_scale = train_scale
        self.look_back_window = look_back_window
        self.shortest_seq = shortest_seq
        self.future_window = future_window
        self.num_stocks = num_stocks

    def get_returns(self, stock_dict: dict) -> List[np.ndarray]:
        return_matrix_list = []

        for i in range(
            self.train_scale + self.look_back_window + 1,
            self.shortest_seq,
            self.future_window,
        ):
            if i + self.future_window > self.shortest_seq:
                break

            return_matrix = np.zeros((self.future_window, self.num_stocks))

            for j, (_, stock_df) in enumerate(stock_dict.items()):
                return_matrix[:, j] = stock_df.iloc[i : i + self.future_window][
                    "ret"
                ].values

            return_matrix_list.append(return_matrix)

        return return_matrix_list

    def form_portfolio(
        self,
        y_preds: List[np.ndarray],
        stock_dict: dict,
        long_num: int,
        short_num: int,
    ) -> Tuple[np.ndarray, ...]:
        """
        Form the portfolio using the predicted alphas and stock returns

        # TODO: Need to figure out how to handle multiple alphas

        Args:
            y_preds (List[np.ndarray]): List of predicted alphas, each element in a list has shape (n_stocks, n_alphas)
            stock_dict (dict): Dictionary containing stock dataframes
            long_num (int): Number of long positions
            short_num (int): Number of short positions

        Returns:
            Tuple[np.ndarray, ...]: Tuple of returns for long, short, benchmark, and portfolio
        """
        sorted_pred_indices = []

        for i in range(len(y_preds)):
            if y_preds[i].shape[-1] == 1:
                sorted_pred_indices.append(np.argsort(y_preds[i][:, 0]))
            else:
                multi_sort = []
                for j in range(y_preds[i].shape[-1]):
                    multi_sort.append(np.argsort(y_preds[i][:, j]))

                sorted_pred_indices.append(multi_sort)

        return_matrix_list = self.get_returns(stock_dict)

        n_days = len(return_matrix_list)
        long_returns = np.zeros((n_days, self.future_window, long_num))
        short_returns = np.zeros((n_days, self.future_window, short_num))
        bench_returns = np.zeros((n_days, self.future_window, self.num_stocks))

        for day in range(n_days):
            for alpha in range(len(y_preds[day])):
                top_k_indices = sorted_pred_indices[day][alpha][-long_num:]
                bottom_k_indices = sorted_pred_indices[day][alpha][:short_num]
                bench_indices = sorted_pred_indices[day][alpha]

                # FIXME: Need to figure out how to handle multiple alphas, return array need to be 4 dim
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
