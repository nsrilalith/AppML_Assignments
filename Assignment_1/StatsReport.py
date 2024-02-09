import pandas as pd


class StatsReport:
    def __init__(self):
        self.data_stats_df = pd.DataFrame()
        self.data_stats_df['Data_Stats'] = ['type', 'cardinality', 'mean', 'median', 'n_at_median', 'mode', 'n_at_mode',
                                            'stddev', 'min', 'max', 'n_rows', 'n_zero', 'n_ques', 'n_missing']
        pass

    def addCol(self, colName, column):
        try:
            mean_value = column.mean()
            median_value = column.median()
            std_dev_value = column.std()
            min_value = column.min()
            max_value = column.max()
        except TypeError:
            mean_value = "N/A"
            median_value = "N/A"
            std_dev_value = "N/A"
            min_value = "N/A"
            max_value = "N/A"
        mode_value = column.mode().iloc[0] if not column.mode().empty else "N/A"
        n_at_mode = (column == mode_value).sum()
        n_at_median = (column == median_value).sum()
        n_zeros = (column == 0).sum()
        n_ques = (column == "?").sum()
        n_missing = column.isna().sum()
        self.data_stats_df[colName] = [column.dtype, column.nunique(), mean_value, median_value, n_at_median,
                                       mode_value, n_at_mode, std_dev_value, min_value, max_value, len(column), n_zeros,
                                       n_ques, n_missing]

    def to_String(self):
        return self.data_stats_df.to_string()

    def write_to_file(self, filepath):
        self.data_stats_df.to_excel(filepath, index=False, engine='openpyxl')
