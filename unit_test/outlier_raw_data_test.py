import unittest
from pathlib import Path
import pandas as pd
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import utils.outliers_helpers as ohp
import utils.changepoint_helpers as chp

class OutlierTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        csv_path = PROJECT_ROOT / "data" / "Bangladesh" / "target" / "raw_data.csv"
        ts_file = pd.read_csv(csv_path)
        ts_file['Date'] = pd.to_datetime(ts_file['Date'], format='%d/%m/%Y')
        # Correct years that are mistakenly in the 2000s
        ts_file['Date'] = ts_file['Date'].apply(lambda x: x.replace(year=x.year - 100) if x.year > 2025 else x)
        ts_file = ts_file.set_index('Date')
        cls.df = ts_file

    def test_shape(self):
        self.assertEqual(self.df.shape, (708, 1250))

    def test_detect_outlier_SY101(self):
        ts_raw = self.df['SY101']
        
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())

        self.assertEqual(outlier_found, True)
        self.assertFalse(np.isnan(ts_raw['1997-03-15']))
        self.assertTrue(np.isnan(ts_cleaned['1997-03-15']))

    def test_detect_outlier_BO008(self):
        ts_raw = self.df['BO008']

        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        self.assertEqual(outlier_found, True)
        self.assertFalse(np.isnan(ts_raw['1997-05-15']))
        self.assertTrue(np.isnan(ts_cleaned['1997-05-15']))

    def test_detect_outlier_RJ011_last_value(self):
        ts_raw = self.df['RJ011']

        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        self.assertEqual(outlier_found, True)
        self.assertFalse(np.isnan(ts_raw['2018-09-15']))
        self.assertTrue(np.isnan(ts_cleaned['2018-09-15']))

    def test_detect_no_outlier_wellchange_present(self):
        ts_raw = self.df['DH103S']

        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        self.assertEqual(outlier_found, False)

    def test_detect_outlier_after_wellchange_DH103S(self):
        ts_raw = self.df['DH103S']

        ts, filtered_breaks = chp.detect_well_change(ts_raw.copy())
        self.assertEqual(len(filtered_breaks), 1)
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw[:filtered_breaks[0]].copy())
        self.assertEqual(outlier_found, True)

    def test_detect_no_outlier(self):
        ts_raw = self.df['FA016']

        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        self.assertEqual(outlier_found, False)

if __name__ == "__main__":
    unittest.main()
